# A dictionary of our pre-defined investment strategies.
# The keys are identifiers, and the values contain metadata and rules.
# The rules map a database column to a tuple of (operator, value).
SCREENER_STRATEGIES = {
    "QUALITY_INVESTING": {
        "name": "Quality Investing (Beginner Friendly)",
        "description": "Focuses on large, stable, profitable companies with low debt. Best for beginners, long-term investors, or those seeking lower-risk options.",
        "rules": {
            "mcap": (">", 50000),          # Market Cap > 50,000 Cr (Large Cap)
            "debt_to_equity": ("<", 1.0),    # Low debt
            "roce": (">", 15.0),            # Good return on capital employed
            "pat_growth_3y": (">", 10.0)    # Consistent profit after tax growth
        }
    },
    "VALUE_INVESTING": {
        "name": "Value Investing",
        "description": "Identifies companies that may be trading below their intrinsic value, indicated by low valuation multiples. Suitable for investors looking for potentially undervalued stocks.",
        "rules": {
            "ttmpe": ("<", 20),             # Low Price-to-Earnings
            "price_bv": ("<", 3),           # Low Price-to-Book Value
            "debt_to_equity": ("<", 1.5)
        }
    },
    "GROWTH_INVESTING": {
        "name": "Growth Investing",
        "description": "Targets companies demonstrating strong, above-average growth in sales and profits, suggesting strong market demand and expansion. Suitable for investors with a higher risk tolerance.",
        "rules": {
            "sales_growth_3y": (">", 20.0), # High sales growth
            "pat_growth_3y": (">", 20.0),   # High profit growth
            "roce": (">", 15.0)             # Must also be efficient
        }
    },
    "DIVIDEND_YIELD": {
        "name": "High Dividend Yield",
        "description": "Finds companies that return a significant portion of their earnings to shareholders as dividends. Best for income-focused investors.",
        "rules": {
            "dividend_yield": (">", 4.0),   # High dividend yield
            "mcap": (">", 10000),           # Filter for reasonably sized companies
            "ttmpe": ("<", 30)              # Avoid potential value traps
        }
    }
}