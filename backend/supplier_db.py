"""
Supplier Database

SQLite cache for supplier research results.
Stores suppliers, products, and search results with TTL-based expiration.
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Default cache duration
DEFAULT_CACHE_HOURS = 24


class SupplierDB:
    """
    SQLite database for caching supplier research results.
    Enables fast re-queries without hitting external APIs.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Args:
            db_path: Path to SQLite database file.
                     Defaults to data/supplier_cache.db
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent / "data" / "supplier_cache.db"
        else:
            db_path = Path(db_path)

        # Ensure directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db_path = str(db_path)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_connection()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS suppliers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    url TEXT UNIQUE NOT NULL,
                    domain TEXT,
                    verified_at TIMESTAMP,
                    is_accessible BOOLEAN DEFAULT 1,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS products (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    supplier_id INTEGER,
                    product_name TEXT NOT NULL,
                    product_url TEXT,
                    price REAL,
                    size REAL,
                    unit TEXT,
                    price_per_oz REAL,
                    in_stock BOOLEAN,
                    shipping_note TEXT,
                    moq INTEGER,
                    raw_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (supplier_id) REFERENCES suppliers(id)
                );

                CREATE TABLE IF NOT EXISTS search_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    normalized_query TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_suppliers_domain ON suppliers(domain);
                CREATE INDEX IF NOT EXISTS idx_products_supplier ON products(supplier_id);
                CREATE INDEX IF NOT EXISTS idx_search_query ON search_cache(normalized_query);
                CREATE INDEX IF NOT EXISTS idx_search_expires ON search_cache(expires_at);
            """)
            conn.commit()
            logger.info(f"Supplier database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize supplier database: {e}")
            raise
        finally:
            conn.close()

    def _normalize_query(self, query: str) -> str:
        """Normalize query for cache lookup."""
        return query.lower().strip()

    # =========================================================================
    # Supplier operations
    # =========================================================================

    def save_supplier(
        self,
        name: str,
        url: str,
        domain: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> int:
        """
        Save or update a supplier.
        Returns supplier ID.
        """
        if domain is None:
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc.replace("www.", "")
            except Exception:
                domain = None

        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                INSERT INTO suppliers (name, url, domain, notes, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(url) DO UPDATE SET
                    name = excluded.name,
                    domain = excluded.domain,
                    notes = excluded.notes,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id
            """, (name, url, domain, notes))
            row = cursor.fetchone()
            conn.commit()
            return row[0]
        finally:
            conn.close()

    def get_supplier_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Get supplier by URL."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM suppliers WHERE url = ?",
                (url,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_supplier_by_domain(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get supplier by domain (e.g., 'edenbotanicals.com')."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM suppliers WHERE domain LIKE ?",
                (f"%{domain}%",)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def mark_supplier_inaccessible(self, supplier_id: int):
        """Mark a supplier as inaccessible (failed HEAD request)."""
        conn = self._get_connection()
        try:
            conn.execute("""
                UPDATE suppliers
                SET is_accessible = 0, verified_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (supplier_id,))
            conn.commit()
        finally:
            conn.close()

    # =========================================================================
    # Product operations
    # =========================================================================

    def save_product(
        self,
        supplier_id: int,
        product_name: str,
        product_url: str,
        price: Optional[float] = None,
        size: Optional[float] = None,
        unit: Optional[str] = None,
        price_per_oz: Optional[float] = None,
        in_stock: Optional[bool] = None,
        shipping_note: Optional[str] = None,
        moq: Optional[int] = None,
        raw_data: Optional[dict] = None,
    ) -> int:
        """
        Save a product for a supplier.
        Returns product ID.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                INSERT INTO products (
                    supplier_id, product_name, product_url,
                    price, size, unit, price_per_oz,
                    in_stock, shipping_note, moq, raw_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
            """, (
                supplier_id, product_name, product_url,
                price, size, unit, price_per_oz,
                in_stock, shipping_note, moq,
                json.dumps(raw_data) if raw_data else None
            ))
            row = cursor.fetchone()
            conn.commit()
            return row[0]
        finally:
            conn.close()

    def get_products_by_supplier(self, supplier_id: int) -> List[Dict[str, Any]]:
        """Get all products for a supplier."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM products WHERE supplier_id = ? ORDER BY created_at DESC",
                (supplier_id,)
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    # =========================================================================
    # Search cache operations
    # =========================================================================

    async def get_cached_results(
        self,
        query: str,
        max_age_hours: int = DEFAULT_CACHE_HOURS,
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached search results for a query.
        Returns None if no valid cache exists.
        """
        normalized = self._normalize_query(query)
        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                SELECT result_json FROM search_cache
                WHERE normalized_query = ?
                AND expires_at > CURRENT_TIMESTAMP
                ORDER BY created_at DESC
                LIMIT 1
            """, (normalized,))
            row = cursor.fetchone()
            if row:
                logger.info(f"Cache hit for query: {query}")
                return json.loads(row[0])
            return None
        finally:
            conn.close()

    async def cache_results(
        self,
        query: str,
        products: List,  # List of ProductData
        cache_hours: int = DEFAULT_CACHE_HOURS,
    ):
        """
        Cache search results for a query.
        """
        normalized = self._normalize_query(query)
        expires_at = datetime.utcnow() + timedelta(hours=cache_hours)

        # Convert products to JSON-serializable format
        result = {
            "query": query,
            "products": [
                {
                    "supplier_name": p.supplier_name,
                    "product_name": p.product_name,
                    "url": p.url,
                    "price": p.price,
                    "size": p.size,
                    "unit": p.unit,
                    "price_per_oz": p.price_per_oz,
                    "in_stock": p.in_stock,
                }
                for p in products
            ],
            "cached_at": datetime.utcnow().isoformat(),
        }

        # Also save suppliers and products to main tables
        for p in products:
            try:
                supplier_id = self.save_supplier(p.supplier_name, p.url)
                self.save_product(
                    supplier_id=supplier_id,
                    product_name=p.product_name,
                    product_url=p.url,
                    price=p.price,
                    size=p.size,
                    unit=p.unit,
                    price_per_oz=p.price_per_oz,
                    in_stock=p.in_stock,
                )
            except Exception as e:
                logger.warning(f"Failed to save product to DB: {e}")

        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO search_cache (query, normalized_query, result_json, expires_at)
                VALUES (?, ?, ?, ?)
            """, (query, normalized, json.dumps(result), expires_at))
            conn.commit()
            logger.info(f"Cached results for query: {query}")
        finally:
            conn.close()

    def clear_expired_cache(self):
        """Remove expired cache entries."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                DELETE FROM search_cache
                WHERE expires_at < CURRENT_TIMESTAMP
            """)
            deleted = cursor.rowcount
            conn.commit()
            if deleted > 0:
                logger.info(f"Cleared {deleted} expired cache entries")
            return deleted
        finally:
            conn.close()

    # =========================================================================
    # Utility methods
    # =========================================================================

    def get_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        conn = self._get_connection()
        try:
            stats = {}
            for table in ["suppliers", "products", "search_cache"]:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]
            return stats
        finally:
            conn.close()


# Singleton instance
_supplier_db: Optional[SupplierDB] = None


def get_supplier_db(db_path: Optional[str] = None) -> SupplierDB:
    """Get or create the supplier database singleton."""
    global _supplier_db
    if _supplier_db is None:
        _supplier_db = SupplierDB(db_path)
    return _supplier_db
