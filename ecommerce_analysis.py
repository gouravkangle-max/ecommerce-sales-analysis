"""
E-commerce Sales & Customer Analysis (Olist dataset)
====================================================
One script that runs the whole project on your laptop:
  1. Load and clean the Olist CSV files
  2. Build a master table
  3. Run SQL analysis (SQLite, built into Python - no database install needed)
  4. Create EDA charts
  5. RFM customer segmentation
  6. Auto-generate an insights report

SETUP (one time)
  pip install pandas numpy matplotlib seaborn

DATA
  Download "Brazilian E-Commerce Public Dataset by Olist" from Kaggle,
  unzip it, and put the CSV files in a folder, e.g.  data/raw/

RUN
  python ecommerce_analysis.py --data data/raw
  (or just: python ecommerce_analysis.py   if the CSVs are in data/raw)

OUTPUT (created in ./output)
  cleaned/master_orders.csv, cleaned/rfm_segments.csv
  sql_results/*.csv        (result of each SQL query)
  charts/*.png             (charts for your README and dashboard)
  insights_report.md       (findings with real numbers)
  analysis.db              (SQLite database you can open in DB Browser)
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save charts to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")

REQUIRED_FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "trans": "product_category_name_translation.csv",
}

DATE_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


def step(msg):
    print(f"\n=== {msg} ===")


# --------------------------------------------------------------------------
# STEP 1: LOAD
# --------------------------------------------------------------------------
def load_data(data_dir: Path) -> dict:
    step("Step 1: Loading data")
    missing = [f for f in REQUIRED_FILES.values() if not (data_dir / f).exists()]
    if missing:
        print(f"ERROR: these files were not found in '{data_dir}':")
        for f in missing:
            print("  -", f)
        print("\nDownload the Olist dataset from Kaggle, unzip it, and point --data to that folder.")
        sys.exit(1)

    tables = {}
    for key, fname in REQUIRED_FILES.items():
        tables[key] = pd.read_csv(data_dir / fname)
        print(f"  loaded {fname:<45} {tables[key].shape}")
    for c in DATE_COLS:
        tables["orders"][c] = pd.to_datetime(tables["orders"][c], errors="coerce")
    return tables


# --------------------------------------------------------------------------
# STEP 2: CLEAN + MASTER TABLE
# --------------------------------------------------------------------------
def clean_and_merge(t: dict):
    step("Step 2: Cleaning and building the master table")
    orders, items, payments = t["orders"], t["items"], t["payments"]
    reviews, customers, products, trans = t["reviews"], t["customers"], t["products"], t["trans"]

    # Data quality summary (print so you can screenshot it for your README)
    for name in ["orders", "items", "payments", "reviews", "customers", "products"]:
        d = t[name]
        print(f"  {name:<10} rows={len(d):>7}  null cells={int(d.isnull().sum().sum()):>7}  duplicate rows={int(d.duplicated().sum())}")

    # Keep delivered orders with a delivery date
    before = len(orders)
    orders = orders[orders["order_status"] == "delivered"].dropna(subset=["order_delivered_customer_date"]).copy()
    print(f"  orders kept after filtering delivered: {len(orders)} of {before}")

    # One review per order
    reviews = reviews.drop_duplicates(subset="order_id")

    # Categories: fill missing and translate to English
    products = products.copy()
    products["product_category_name"] = products["product_category_name"].fillna("unknown")
    products = products.merge(trans, on="product_category_name", how="left")
    products["category"] = products["product_category_name_english"].fillna(products["product_category_name"])

    # Payments: one row per order (avoid double counting later)
    order_pay = (
        payments.groupby("order_id")
        .agg(payment_value=("payment_value", "sum"), payment_type=("payment_type", "first"))
        .reset_index()
    )

    master = (
        orders.merge(customers, on="customer_id", how="left")
        .merge(items, on="order_id", how="inner")
        .merge(products[["product_id", "category"]], on="product_id", how="left")
        .merge(reviews[["order_id", "review_score"]], on="order_id", how="left")
    )
    master["category"] = master["category"].fillna("unknown")

    # Feature engineering
    master["order_month"] = master["order_purchase_timestamp"].dt.to_period("M").astype(str)
    master["order_year"] = master["order_purchase_timestamp"].dt.year
    master["delivery_days"] = (
        master["order_delivered_customer_date"] - master["order_purchase_timestamp"]
    ).dt.days
    master["delay_days"] = (
        master["order_delivered_customer_date"] - master["order_estimated_delivery_date"]
    ).dt.days
    master["is_late"] = (master["delay_days"] > 0).astype(int)
    master["revenue"] = master["price"] + master["freight_value"]

    # Drop the 2016 tail / partial months if you want; we keep all data here.
    print(f"  master table: {master.shape[0]} rows x {master.shape[1]} columns")
    print(f"  date range: {master['order_purchase_timestamp'].min().date()} to {master['order_purchase_timestamp'].max().date()}")
    return master, order_pay


# --------------------------------------------------------------------------
# STEP 3: SQL ANALYSIS
# --------------------------------------------------------------------------
SQL_QUERIES = {
    "q1_monthly_revenue_orders": """
        SELECT order_month AS month,
               COUNT(DISTINCT order_id) AS orders,
               ROUND(SUM(revenue), 2)   AS revenue
        FROM master_orders
        GROUP BY order_month
        ORDER BY order_month;
    """,
    "q2_mom_growth": """
        WITH monthly AS (
            SELECT order_month AS month, SUM(revenue) AS revenue
            FROM master_orders
            GROUP BY order_month
        )
        SELECT month,
               ROUND(revenue, 2) AS revenue,
               ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
                     / LAG(revenue) OVER (ORDER BY month), 1) AS mom_growth_pct
        FROM monthly
        ORDER BY month;
    """,
    "q3_top_categories": """
        SELECT category,
               ROUND(SUM(revenue), 0) AS revenue,
               ROUND(100.0 * SUM(revenue) / SUM(SUM(revenue)) OVER (), 1) AS pct_of_total
        FROM master_orders
        GROUP BY category
        ORDER BY revenue DESC
        LIMIT 10;
    """,
    "q4_revenue_by_state": """
        SELECT customer_state,
               COUNT(DISTINCT order_id) AS orders,
               ROUND(SUM(revenue), 0) AS revenue,
               ROUND(100.0 * SUM(revenue) / SUM(SUM(revenue)) OVER (), 1) AS pct_of_total
        FROM master_orders
        GROUP BY customer_state
        ORDER BY revenue DESC;
    """,
    "q5_repeat_vs_onetime": """
        WITH cust AS (
            SELECT customer_unique_id, COUNT(DISTINCT order_id) AS num_orders
            FROM master_orders
            GROUP BY customer_unique_id
        )
        SELECT CASE WHEN num_orders = 1 THEN 'One-time' ELSE 'Repeat' END AS customer_type,
               COUNT(*) AS customers,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
        FROM cust
        GROUP BY customer_type;
    """,
    "q6_late_delivery_vs_review": """
        SELECT CASE WHEN is_late = 1 THEN 'Late' ELSE 'On time' END AS delivery,
               COUNT(DISTINCT order_id) AS orders,
               ROUND(AVG(review_score), 2) AS avg_review
        FROM (SELECT DISTINCT order_id, is_late, review_score FROM master_orders)
        WHERE review_score IS NOT NULL
        GROUP BY delivery;
    """,
    "q7_payment_methods": """
        SELECT payment_type,
               COUNT(*) AS orders,
               ROUND(SUM(payment_value), 0) AS total_value,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_orders
        FROM order_payments
        WHERE order_id IN (SELECT DISTINCT order_id FROM master_orders)
        GROUP BY payment_type
        ORDER BY orders DESC;
    """,
    "q8_top3_categories_per_state": """
        SELECT * FROM (
            SELECT customer_state, category,
                   ROUND(SUM(revenue), 0) AS revenue,
                   RANK() OVER (PARTITION BY customer_state ORDER BY SUM(revenue) DESC) AS rnk
            FROM master_orders
            GROUP BY customer_state, category
        )
        WHERE rnk <= 3
        ORDER BY customer_state, rnk;
    """,
}


def run_sql(master, order_pay, out: Path) -> dict:
    step("Step 3: SQL analysis (SQLite)")
    db_path = out / "analysis.db"
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)

    m = master.copy()
    for c in m.select_dtypes(include=["datetime64[ns]"]).columns:
        m[c] = m[c].dt.strftime("%Y-%m-%d %H:%M:%S")
    m.to_sql("master_orders", con, index=False)
    order_pay.to_sql("order_payments", con, index=False)

    (out / "sql_results").mkdir(exist_ok=True)
    (out / "analysis_queries.sql").write_text(
        "\n\n".join(f"-- {name}\n{q.strip()}" for name, q in SQL_QUERIES.items()), encoding="utf-8"
    )

    results = {}
    for name, q in SQL_QUERIES.items():
        res = pd.read_sql_query(q, con)
        res.to_csv(out / "sql_results" / f"{name}.csv", index=False)
        results[name] = res
        print(f"\n-- {name}")
        print(res.head(8).to_string(index=False))
    con.close()
    return results


# --------------------------------------------------------------------------
# STEP 4: EDA CHARTS
# --------------------------------------------------------------------------
def make_charts(master, out: Path):
    step("Step 4: Creating charts")
    cdir = out / "charts"
    cdir.mkdir(exist_ok=True)

    def save(name):
        plt.tight_layout()
        plt.savefig(cdir / name, dpi=150)
        plt.close()
        print("  saved", name)

    # 1. Monthly revenue
    monthly = master.groupby("order_month")["revenue"].sum()
    plt.figure(figsize=(12, 5))
    monthly.plot(marker="o", color="#2a6fdb")
    plt.title("Monthly Revenue")
    plt.ylabel("Revenue")
    plt.xlabel("")
    plt.xticks(rotation=60)
    save("01_monthly_revenue.png")

    # 2. Monthly orders
    mo = master.groupby("order_month")["order_id"].nunique()
    plt.figure(figsize=(12, 5))
    mo.plot(kind="bar", color="#4aa37a")
    plt.title("Monthly Orders")
    plt.xlabel("")
    plt.xticks(rotation=60)
    save("02_monthly_orders.png")

    # 3. Top 10 categories
    top = master.groupby("category")["revenue"].sum().nlargest(10).sort_values()
    plt.figure(figsize=(9, 5))
    top.plot(kind="barh", color="#e08a2e")
    plt.title("Top 10 Categories by Revenue")
    plt.xlabel("Revenue")
    plt.ylabel("")
    save("03_top_categories.png")

    # 4. Top 10 states
    st = master.groupby("customer_state")["revenue"].sum().nlargest(10).sort_values()
    plt.figure(figsize=(9, 5))
    st.plot(kind="barh", color="#8a5cd0")
    plt.title("Top 10 States by Revenue")
    plt.xlabel("Revenue")
    plt.ylabel("")
    save("04_top_states.png")

    # 5. Review score: late vs on-time (order level)
    od = master.drop_duplicates("order_id").dropna(subset=["review_score"])
    plt.figure(figsize=(6, 5))
    sns.barplot(data=od, x="is_late", y="review_score", hue="is_late", palette=["#4aa37a", "#d9534f"], legend=False)
    plt.xticks([0, 1], ["On time", "Late"])
    plt.xlabel("")
    plt.ylabel("Average review score")
    plt.title("Late Delivery vs Review Score")
    save("05_late_vs_review.png")

    # 6. Delivery days distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(od["delivery_days"].clip(upper=60), bins=30, color="#2a6fdb")
    plt.title("Delivery Time Distribution (days, capped at 60)")
    plt.xlabel("Days from purchase to delivery")
    save("06_delivery_days.png")

    # 7. Payment methods
    pay = od.merge(master[["order_id"]].drop_duplicates(), on="order_id")
    # use order-level payment table stored in master? payment_type is not in master, so skip here
    # (payment chart is created in main() from the SQL result)

    # 8. Correlation heatmap
    cols = ["price", "freight_value", "delivery_days", "delay_days", "review_score"]
    plt.figure(figsize=(7, 5))
    sns.heatmap(master[cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    save("08_correlation.png")


# --------------------------------------------------------------------------
# STEP 5: RFM
# --------------------------------------------------------------------------
def rfm_segmentation(master, out: Path):
    step("Step 5: RFM customer segmentation")
    snapshot = master["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

    rfm = (
        master.groupby("customer_unique_id")
        .agg(
            recency=("order_purchase_timestamp", lambda x: (snapshot - x.max()).days),
            frequency=("order_id", "nunique"),
            monetary=("revenue", "sum"),
        )
        .reset_index()
    )

    # rank(method="first") avoids duplicate-bin errors (most customers buy only once)
    rfm["R"] = pd.qcut(rfm["recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["score"] = rfm["R"] + rfm["F"] + rfm["M"]

    def segment(s):
        if s >= 13:
            return "Champions"
        if s >= 10:
            return "Loyal"
        if s >= 7:
            return "Potential"
        if s >= 5:
            return "At Risk"
        return "Lost"

    rfm["segment"] = rfm["score"].apply(segment)
    (out / "cleaned").mkdir(exist_ok=True)
    rfm.to_csv(out / "cleaned" / "rfm_segments.csv", index=False)

    summary = (
        rfm.groupby("segment")
        .agg(customers=("customer_unique_id", "count"),
             avg_recency=("recency", "mean"),
             avg_frequency=("frequency", "mean"),
             avg_monetary=("monetary", "mean"),
             total_revenue=("monetary", "sum"))
        .round(1)
        .sort_values("total_revenue", ascending=False)
    )
    summary["revenue_share_pct"] = (100 * summary["total_revenue"] / summary["total_revenue"].sum()).round(1)
    print(summary.to_string())
    summary.to_csv(out / "sql_results" / "rfm_summary.csv")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    summary["customers"].plot(kind="bar", ax=ax[0], color="#2a6fdb")
    ax[0].set_title("Customers per Segment")
    ax[0].set_xlabel("")
    summary["revenue_share_pct"].plot(kind="bar", ax=ax[1], color="#e08a2e")
    ax[1].set_title("Revenue Share (%) per Segment")
    ax[1].set_xlabel("")
    for a in ax:
        a.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(out / "charts" / "09_rfm_segments.png", dpi=150)
    plt.close()
    print("  saved 09_rfm_segments.png")
    return summary


# --------------------------------------------------------------------------
# STEP 6: INSIGHTS REPORT
# --------------------------------------------------------------------------
def write_report(master, results, rfm_summary, out: Path):
    step("Step 6: Writing insights report")
    r = results
    monthly = r["q1_monthly_revenue_orders"]
    peak = monthly.loc[monthly["revenue"].idxmax()]
    top_cat = r["q3_top_categories"]
    top3_share = top_cat["pct_of_total"].head(3).sum()
    states = r["q4_revenue_by_state"]
    top_states = states.head(3)
    rep = r["q5_repeat_vs_onetime"].set_index("customer_type")
    one_pct = rep.loc["One-time", "pct"] if "One-time" in rep.index else float("nan")
    late = r["q6_late_delivery_vs_review"].set_index("delivery")
    pay = r["q7_payment_methods"]

    total_rev = master["revenue"].sum()
    total_orders = master["order_id"].nunique()
    aov = total_rev / total_orders
    late_rate = master.drop_duplicates("order_id")["is_late"].mean() * 100

    lines = [
        "# Insights Report: E-commerce Sales & Customer Analysis",
        "",
        "## Headline KPIs",
        f"- Total revenue: {total_rev:,.0f}",
        f"- Total delivered orders: {total_orders:,}",
        f"- Average order value: {aov:,.2f}",
        f"- Late delivery rate: {late_rate:.1f}%",
        "",
        "## Findings",
        "",
        "### 1. Revenue trend",
        f"- Evidence: revenue peaked in {peak['month']} at {peak['revenue']:,.0f} across {int(peak['orders']):,} orders.",
        "- Recommendation: plan inventory, staffing and promotions ahead of peak months.",
        "",
        "### 2. A few categories dominate",
        f"- Evidence: the top 3 categories ({', '.join(top_cat['category'].head(3))}) generate {top3_share:.1f}% of revenue.",
        "- Recommendation: protect stock and quality in these categories, and test growth in smaller ones.",
        "",
        "### 3. Revenue is concentrated geographically",
        f"- Evidence: {', '.join(top_states['customer_state'])} contribute {top_states['pct_of_total'].sum():.1f}% of revenue.",
        "- Recommendation: improve logistics and marketing in under-served states.",
        "",
        "### 4. Most customers buy only once",
        f"- Evidence: {one_pct:.1f}% of customers are one-time buyers.",
        "- Recommendation: post-purchase emails, loyalty points and win-back offers to lift repeat purchases.",
        "",
        "### 5. Late deliveries hurt satisfaction",
    ]
    if "Late" in late.index and "On time" in late.index:
        diff = late.loc["On time", "avg_review"] - late.loc["Late", "avg_review"]
        lines.append(
            f"- Evidence: average review is {late.loc['On time','avg_review']} for on-time orders vs "
            f"{late.loc['Late','avg_review']} for late orders (a gap of {diff:.2f} points)."
        )
    lines += [
        "- Recommendation: work with carriers in states with the highest late rates and set more realistic delivery estimates.",
        "",
        "### 6. Payment behaviour",
        f"- Evidence: {pay.iloc[0]['payment_type']} is the most used payment method ({pay.iloc[0]['pct_orders']}% of orders).",
        "- Recommendation: make sure the top payment methods are smooth and consider offers on preferred methods.",
        "",
        "## Customer segments (RFM)",
        rfm_summary[["customers", "revenue_share_pct"]].to_markdown()
        if _has_tabulate()
        else rfm_summary[["customers", "revenue_share_pct"]].to_string(),
        "",
        "- Champions / Loyal: reward and retain with early access and loyalty perks.",
        "- Potential: nudge toward a second purchase with personalised offers.",
        "- At Risk / Lost: run win-back campaigns.",
    ]
    (out / "insights_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("  saved insights_report.md")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Olist e-commerce analysis")
    ap.add_argument("--data", default="data/raw", help="folder containing the Olist CSV files")
    ap.add_argument("--out", default="output", help="folder for results")
    args = ap.parse_args()

    data_dir, out = Path(args.data), Path(args.out)
    (out / "cleaned").mkdir(parents=True, exist_ok=True)

    tables = load_data(data_dir)
    master, order_pay = clean_and_merge(tables)
    master.to_csv(out / "cleaned" / "master_orders.csv", index=False)

    results = run_sql(master, order_pay, out)
    make_charts(master, out)

    # Payment methods chart from SQL result
    pay = results["q7_payment_methods"]
    plt.figure(figsize=(7, 5))
    plt.pie(pay["orders"], labels=pay["payment_type"], autopct="%1.1f%%", startangle=90)
    plt.title("Orders by Payment Method")
    plt.tight_layout()
    plt.savefig(out / "charts" / "07_payment_methods.png", dpi=150)
    plt.close()
    print("  saved 07_payment_methods.png")

    rfm_summary = rfm_segmentation(master, out)
    write_report(master, results, rfm_summary, out)

    step("DONE")
    print(f"All results are in: {out.resolve()}")
    print("Next: import output/cleaned/master_orders.csv into Power BI or Tableau for the dashboard.")


if __name__ == "__main__":
    main()
