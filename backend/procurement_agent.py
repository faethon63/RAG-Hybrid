"""
Procurement Research Agent

Full workflow for finding suppliers and extracting product data:
1. Query Understanding - Extract product name, quantity, use case
2. Supplier Discovery - Use Perplexity to find potential suppliers
3. Site Verification - Check which sites are accessible
4. On-Site Search - Use Playwright to navigate and search within supplier sites
5. Data Extraction - Extract structured product data (price, size, stock)
6. Calculations - Compute $/oz for comparison
7. Format Output - Present comparison table
"""

import asyncio
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ProductData:
    """Structured product data extracted from a supplier page."""
    supplier_name: str
    product_name: str
    url: str
    price: Optional[float] = None
    size: Optional[float] = None
    unit: Optional[str] = None  # ml, oz, g, kg, lb
    price_per_oz: Optional[float] = None
    in_stock: Optional[bool] = None
    shipping_note: Optional[str] = None
    moq: Optional[int] = None  # Minimum order quantity


class ProcurementAgent:
    """
    Orchestrates supplier research using Perplexity for discovery
    and Playwright for on-site navigation and search.
    """

    # Common search box selectors (tried in order)
    SEARCH_SELECTORS = [
        'input[type="search"]',
        'input[name="q"]',
        'input[name="s"]',
        'input[name="search"]',
        'input[name="query"]',
        'input[placeholder*="search" i]',
        'input[placeholder*="Search" i]',
        'input[aria-label*="search" i]',
        '.search-input',
        '.search-field',
        '#search',
        '#search-input',
        '#searchbox',
        '[data-testid="search-input"]',
    ]

    # Unit conversion factors (to ounces)
    UNIT_TO_OZ = {
        'oz': 1.0,
        'fl oz': 1.0,
        'floz': 1.0,
        'ml': 0.033814,
        'l': 33.814,
        'liter': 33.814,
        'litre': 33.814,
        'g': 0.035274,
        'gram': 0.035274,
        'grams': 0.035274,
        'kg': 35.274,
        'kilogram': 35.274,
        'lb': 16.0,
        'lbs': 16.0,
        'pound': 16.0,
    }

    def __init__(self, perplexity_search=None, supplier_db=None):
        """
        Args:
            perplexity_search: PerplexitySearch instance for supplier discovery
            supplier_db: SupplierDB instance for caching results
        """
        self.perplexity = perplexity_search
        self.supplier_db = supplier_db
        self._http_client: Optional[httpx.AsyncClient] = None

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=10.0)
        return self._http_client

    async def research_product(
        self,
        query: str,
        max_suppliers: int = 5,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Main entry point: Research suppliers for a product.

        Args:
            query: User's query (e.g., "where can I buy alpha cedrene")
            max_suppliers: Maximum number of suppliers to research
            use_cache: Whether to check cache first

        Returns:
            {
                "answer": str,  # Formatted comparison table
                "products": List[ProductData],
                "sources": List[dict],
            }
        """
        logger.info(f"Starting procurement research for: {query}")

        # 1. Extract product name from query
        product_name = self._extract_product_name(query)
        logger.info(f"Extracted product name: {product_name}")

        # 2. Check cache
        if use_cache and self.supplier_db:
            cached = await self.supplier_db.get_cached_results(product_name)
            if cached:
                logger.info(f"Returning cached results for {product_name}")
                return cached

        # 3. Discover suppliers via Perplexity
        suppliers = await self.discover_suppliers(product_name)
        logger.info(f"Discovered {len(suppliers)} potential suppliers")

        if not suppliers:
            return {
                "answer": f"No suppliers found for {product_name}. Try a different search term.",
                "products": [],
                "sources": [],
            }

        # 4. Verify site accessibility (parallel HEAD requests)
        accessible = await self.verify_sites([s["url"] for s in suppliers])
        suppliers = [s for s in suppliers if s["url"] in accessible]
        logger.info(f"{len(suppliers)} suppliers are accessible")

        # 5. Search within each supplier site (parallel Playwright)
        products = await self._search_all_suppliers(suppliers[:max_suppliers], product_name)
        logger.info(f"Found {len(products)} products")

        # 6. Calculate price per oz
        for product in products:
            if product.price and product.size and product.unit:
                product.price_per_oz = self.calculate_price_per_unit(
                    product.price, product.size, product.unit
                )

        # 7. Sort by price per oz (cheapest first)
        products_with_price = [p for p in products if p.price_per_oz]
        products_without_price = [p for p in products if not p.price_per_oz]
        products = sorted(products_with_price, key=lambda p: p.price_per_oz) + products_without_price

        # 8. Cache results
        if self.supplier_db and products:
            await self.supplier_db.cache_results(product_name, products)

        # 9. Format output
        answer = self._format_comparison_table(product_name, products)

        return {
            "answer": answer,
            "products": [self._product_to_dict(p) for p in products],
            "sources": [{"url": p.url, "title": p.supplier_name} for p in products],
        }

    def _extract_product_name(self, query: str) -> str:
        """Extract the product name from a query like 'where can I buy alpha cedrene'."""
        # Remove common query patterns
        patterns = [
            r"where (?:can|do) (?:i|we) (?:buy|find|get|purchase)\s+",
            r"find suppliers? (?:for|of)\s+",
            r"buy\s+",
            r"looking for\s+",
            r"need to (?:find|buy|get)\s+",
            r"search for\s+",
        ]
        result = query.lower()
        for pattern in patterns:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)

        # Remove trailing qualifiers
        result = re.sub(r"\s+(?:online|near me|cheap|best|quality).*$", "", result)

        return result.strip()

    async def discover_suppliers(self, product_name: str) -> List[Dict[str, str]]:
        """
        Use Perplexity to find potential suppliers for a product.

        Returns list of {"name": str, "url": str, "snippet": str}
        """
        if not self.perplexity:
            logger.error("Perplexity search not configured")
            return []

        try:
            # Use focused_search for supplier tables
            result = await self.perplexity.focused_search(
                query=f"buy {product_name} supplier",
                num_results=10,
                recency="month",
            )

            suppliers = []

            # Extract suppliers from citations
            for citation in result.get("citations", []):
                url = citation.get("url", "")
                if url and self._is_likely_supplier(url):
                    suppliers.append({
                        "name": self._extract_supplier_name(url),
                        "url": url,
                        "snippet": citation.get("snippet", ""),
                    })

            # Also try to extract from the answer text (may have URLs in table)
            answer = result.get("answer", "")
            url_pattern = r'https?://[^\s\)"\]]+/[^\s\)"\]]+'
            for url in re.findall(url_pattern, answer):
                url = url.rstrip(".,;")
                if url not in [s["url"] for s in suppliers] and self._is_likely_supplier(url):
                    suppliers.append({
                        "name": self._extract_supplier_name(url),
                        "url": url,
                        "snippet": "",
                    })

            return suppliers[:10]  # Max 10 suppliers

        except Exception as e:
            logger.error(f"Supplier discovery failed: {e}")
            return []

    def _is_likely_supplier(self, url: str) -> bool:
        """Check if URL is likely a product supplier (not a blog, wiki, etc.)."""
        # Exclude non-supplier sites
        excluded = [
            "wikipedia.org", "reddit.com", "quora.com", "pinterest.com",
            "youtube.com", "instagram.com", "facebook.com", "twitter.com",
            "amazon.com", "ebay.com", "walmart.com",  # Major marketplaces (blocked by scrapers)
            "google.com", "bing.com",
        ]
        url_lower = url.lower()
        return not any(ex in url_lower for ex in excluded)

    def _extract_supplier_name(self, url: str) -> str:
        """Extract supplier name from URL."""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            # Remove www. and TLD
            name = domain.replace("www.", "").split(".")[0]
            # Title case
            return name.title().replace("-", " ").replace("_", " ")
        except Exception:
            return url

    async def verify_sites(self, urls: List[str]) -> List[str]:
        """
        Check which URLs are accessible (HEAD request).
        Returns list of accessible URLs.
        """
        client = self._get_http_client()
        accessible = []

        async def check_url(url: str) -> Optional[str]:
            try:
                resp = await client.head(url, follow_redirects=True)
                if resp.status_code < 400:
                    return url
            except Exception as e:
                logger.debug(f"Site not accessible: {url} - {e}")
            return None

        # Check all URLs in parallel
        results = await asyncio.gather(*[check_url(url) for url in urls])
        accessible = [url for url in results if url]

        return accessible

    async def _search_all_suppliers(
        self,
        suppliers: List[Dict[str, str]],
        product_name: str,
    ) -> List[ProductData]:
        """
        Search for product on all supplier sites in parallel.
        Uses Playwright for JS-heavy sites.
        """
        tasks = [
            self.search_on_site(s["url"], s["name"], product_name)
            for s in suppliers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        products = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Supplier search failed: {result}")
            elif result:
                products.append(result)

        return products

    async def search_on_site(
        self,
        url: str,
        supplier_name: str,
        product_name: str,
    ) -> Optional[ProductData]:
        """
        Navigate to supplier site, use their search to find product,
        and extract product data from results.

        Uses Playwright for full browser automation.
        """
        try:
            from playwright.async_api import async_playwright

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                )
                page = await context.new_page()

                # Navigate to site
                logger.info(f"Navigating to {url}")
                await page.goto(url, wait_until="networkidle", timeout=15000)

                # Wait for page to stabilize
                await page.wait_for_timeout(1000)

                # Try to find and use search box
                search_used = False
                for selector in self.SEARCH_SELECTORS:
                    try:
                        search_box = page.locator(selector).first
                        if await search_box.is_visible(timeout=1000):
                            await search_box.fill(product_name)
                            await search_box.press("Enter")
                            await page.wait_for_load_state("networkidle", timeout=10000)
                            search_used = True
                            logger.info(f"Search submitted on {supplier_name} using {selector}")
                            break
                    except Exception:
                        continue

                if not search_used:
                    logger.info(f"No search box found on {supplier_name}, trying URL patterns")
                    # Try common search URL patterns
                    search_urls = [
                        f"{url.rstrip('/')}/search?q={product_name.replace(' ', '+')}",
                        f"{url.rstrip('/')}/search?s={product_name.replace(' ', '+')}",
                        f"{url.rstrip('/')}/?s={product_name.replace(' ', '+')}",
                    ]
                    for search_url in search_urls:
                        try:
                            await page.goto(search_url, wait_until="networkidle", timeout=10000)
                            break
                        except Exception:
                            continue

                # Extract product data from current page
                product_data = await self._extract_product_data(page, supplier_name, url)

                await browser.close()
                return product_data

        except ImportError:
            logger.error("Playwright not installed - run: pip install playwright && playwright install chromium")
            return None
        except Exception as e:
            logger.error(f"Search on {supplier_name} failed: {e}")
            return None

    async def _extract_product_data(
        self,
        page,
        supplier_name: str,
        base_url: str,
    ) -> Optional[ProductData]:
        """
        Extract product data from current page using various strategies.
        """
        try:
            # Get page content
            content = await page.content()
            text = await page.inner_text("body")
            current_url = page.url

            # Try to find price
            price = await self._extract_price(page, text)

            # Try to find size/quantity
            size, unit = await self._extract_size(text)

            # Check stock status
            in_stock = self._check_stock_status(text)

            # Get product name from page title or h1
            try:
                title = await page.title()
                h1 = await page.locator("h1").first.inner_text(timeout=1000)
                product_name = h1 if h1 else title
            except Exception:
                product_name = supplier_name

            if price or size:  # Only return if we found something useful
                return ProductData(
                    supplier_name=supplier_name,
                    product_name=product_name[:100] if product_name else "",
                    url=current_url,
                    price=price,
                    size=size,
                    unit=unit,
                    in_stock=in_stock,
                )

            return None

        except Exception as e:
            logger.warning(f"Data extraction failed: {e}")
            return None

    async def _extract_price(self, page, text: str) -> Optional[float]:
        """Extract price from page."""
        # Try structured data first (JSON-LD)
        try:
            scripts = await page.locator('script[type="application/ld+json"]').all()
            for script in scripts:
                try:
                    import json
                    data = json.loads(await script.inner_text())
                    if isinstance(data, dict):
                        price = data.get("offers", {}).get("price")
                        if price:
                            return float(price)
                except Exception:
                    continue
        except Exception:
            pass

        # Try common price patterns in text
        price_patterns = [
            r'\$(\d+(?:\.\d{2})?)',
            r'USD\s*(\d+(?:\.\d{2})?)',
            r'Price:\s*\$?(\d+(?:\.\d{2})?)',
            r'(\d+(?:\.\d{2})?)\s*USD',
        ]
        for pattern in price_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        return None

    async def _extract_size(self, text: str) -> Tuple[Optional[float], Optional[str]]:
        """Extract size and unit from text."""
        size_patterns = [
            r'(\d+(?:\.\d+)?)\s*(ml|oz|fl oz|g|kg|lb|lbs|liter|litre)',
            r'(\d+(?:\.\d+)?)(ml|oz|g)',
        ]
        for pattern in size_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    size = float(match.group(1))
                    unit = match.group(2).lower()
                    return size, unit
                except ValueError:
                    continue

        return None, None

    def _check_stock_status(self, text: str) -> Optional[bool]:
        """Check if product appears to be in stock."""
        text_lower = text.lower()
        if any(phrase in text_lower for phrase in ["out of stock", "sold out", "unavailable", "back order"]):
            return False
        if any(phrase in text_lower for phrase in ["in stock", "add to cart", "buy now", "available"]):
            return True
        return None

    def calculate_price_per_unit(
        self,
        price: float,
        size: float,
        unit: str,
    ) -> Optional[float]:
        """Calculate price per oz for comparison."""
        unit_lower = unit.lower().strip()
        conversion = self.UNIT_TO_OZ.get(unit_lower)

        if conversion:
            size_in_oz = size * conversion
            if size_in_oz > 0:
                return round(price / size_in_oz, 2)

        return None

    def _format_comparison_table(
        self,
        product_name: str,
        products: List[ProductData],
    ) -> str:
        """Format products as a comparison table."""
        if not products:
            return f"No products found for {product_name}."

        lines = [
            f"**Supplier comparison for {product_name}:**\n",
            "| Supplier | Price | Size | $/oz | Stock | Link |",
            "|----------|-------|------|------|-------|------|",
        ]

        for p in products:
            price_str = f"${p.price:.2f}" if p.price else "N/A"
            size_str = f"{p.size}{p.unit}" if p.size and p.unit else "N/A"
            price_per_oz = f"${p.price_per_oz:.2f}" if p.price_per_oz else "N/A"
            stock = "Yes" if p.in_stock else ("No" if p.in_stock is False else "?")
            link = f"[Link]({p.url})" if p.url else ""

            lines.append(
                f"| {p.supplier_name[:20]} | {price_str} | {size_str} | {price_per_oz} | {stock} | {link} |"
            )

        return "\n".join(lines)

    def _product_to_dict(self, product: ProductData) -> Dict[str, Any]:
        """Convert ProductData to dict for JSON serialization."""
        return {
            "supplier_name": product.supplier_name,
            "product_name": product.product_name,
            "url": product.url,
            "price": product.price,
            "size": product.size,
            "unit": product.unit,
            "price_per_oz": product.price_per_oz,
            "in_stock": product.in_stock,
            "shipping_note": product.shipping_note,
            "moq": product.moq,
        }


# Singleton instance (initialized lazily with dependencies)
_procurement_agent: Optional[ProcurementAgent] = None


def get_procurement_agent(perplexity_search=None, supplier_db=None) -> ProcurementAgent:
    """Get or create the procurement agent singleton."""
    global _procurement_agent
    if _procurement_agent is None:
        _procurement_agent = ProcurementAgent(
            perplexity_search=perplexity_search,
            supplier_db=supplier_db,
        )
    return _procurement_agent
