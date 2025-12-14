# 🛒 Brazil E-Commerce Price Optimization  
### Data Science & Data Analysis Integrated Project  

This repository presents a comprehensive **data-driven project** focused on predicting and analyzing product prices in Brazil’s e-commerce ecosystem.  
The project integrates two analytical perspectives — **Data Science** (predictive modeling) and **Data Analysis** (descriptive insights) — under one business problem:  
> **How can sellers optimize product pricing strategies to stay competitive and profitable across Brazil’s top 10 e-commerce product categories?**

---

## 🎯 Business Context  

Brazil’s online marketplace includes thousands of sellers offering diverse products with significant price variability.  
Setting competitive yet profitable prices remains a major challenge due to differences in:  

- Product characteristics (weight, dimensions, description, photo count)  
- Seller and customer locations (logistics impact)  
- Delivery performance (on-time rate, delay)  
- Customer satisfaction (review scores and sentiment)  

This project uses data from multiple relational tables (orders, products, sellers, payments, and reviews) to build both **predictive models** and **exploratory analyses** that help sellers make informed pricing decisions.

---

## 🧩 Project Objectives  

1. **Price Prediction (Data Science Track)**  
   - Develop a regression model to predict product prices for the top 10 product categories.  
   - Evaluate model performance using **MedAE**, **MAE**, **R²**, and **MAPE**.  
   - Identify which product features most influence pricing (weight, size, description length, etc.).  

2. **Market Insights (Data Analyst Track)**  
   - Explore buyer, seller, and delivery patterns influencing product value.  
   - Analyze sentiment and keyword patterns in customer reviews.  
   - Identify pricing trends, regional differences, and category-level insights.  

3. **Affiliate Program Simulation**  
   - Design a business model where sellers pay a monthly fee for access to **data-driven pricing tools**.  
   - Estimate profit increase through model-driven recommendations.

---

## 📊 Dataset Overview  

Data source: **Brazilian E-Commerce Public Dataset (Olist)**  
Main tables merged using relational keys (`order_id`, `product_id`, `seller_id`, `customer_id`).

| Table | Description |
|--------|-------------|
| `order_items` | Product-level order data (price, freight, seller, product) |
| `orders` | Order lifecycle timestamps and status |
| `products` | Product physical & descriptive attributes |
| `reviews` | Customer feedback, scores, and comments |
| `payments` | Payment type, installments, and value |
| `customers` | Buyer demographic and geographic info |
| `sellers` | Seller location and postal codes |

---

## 🧠 Data Science Pipeline  

1. **Data Merging & Cleaning**  
   - Merged multiple tables into one comprehensive DataFrame.  
   - Handled missing data, kept legitimate outliers (premium items, delivery delays).  

2. **Feature Selection**  
   - Product features: weight, dimensions, description length, photo quantity  
   - Seller features: state, city  
   - Category: product_category_name_english  
   - Target: `price`  

3. **Preprocessing**  
   - Encoding: OneHotEncoder for categorical features  
   - Scaling: RobustScaler for numeric columns  
   - Transformation: Log-transform `price` to handle skewness  

4. **Modeling & Evaluation**  
   - Models: `RandomForest`, `LightGBM`, `XGBoost`, `LinearRegression`, `DummyRegressor`  
   - Cross-validation: 5-fold  
   - Metrics: **MedAE (primary)**, MAE, R², MaxError, MAPE  

5. **Best Model – Tuned RandomForest (Log-Transformed)**  
   - Test results:  
     - MedAE: **2.03**  
     - MAE: **14.71**  
     - R²: **0.81**  
     - MAPE: **11.5%**  
   - Key Predictors: `product_weight_g`, `product_description_lenght`, `product_height_cm`, `seller_state`.

6. **Deployment Potential**  
   - Model saved using `joblib`  
   - Future deployment via Streamlit dashboard for interactive price recommendations.

---

## 📈 Data Analyst Insights  

### 1. Pricing & Profitability  
- Premium products have longer descriptions and fewer but higher-quality images.  
- Freight cost increases proportionally with product weight.  
- Higher prices often correlate with better review scores.

### 2. Delivery Performance  
- Delayed deliveries directly reduce review scores.  
- São Paulo & Rio de Janeiro have the fastest delivery times.  
- Delivery reliability is the strongest driver of customer satisfaction.

### 3. Customer Sentiment (WordCloud Insights)  
Common positive terms: *“entrega rápida”, “ótimo produto”, “recomendo”*  
Common negatives: *“não recebi”, “veio errado”, “faltando”*  
> → Fast, accurate, and well-packaged deliveries earn the best reviews.

### 4. Regional Pricing  
- Northern regions (AC, RO, RN) show higher average prices due to logistics cost.  
- Competitive pricing dominates urban centers like São Paulo and Rio.  

### 5. Strategic Takeaways  
- Accuracy + Fast Delivery = Customer Trust  
- Detailed listings improve conversion for mid to high-end items.  
- Packaging quality is critical for fragile or multi-component goods.  

---

## 💡 Business Integration: Seller Affiliation Program  

| Metric | Without Program | With Program |
|:--|--:|--:|
| Orders | 10,000 | 12,000 |
| Gross Revenue | R$1.65M | R$1.98M |
| Profit Margin | 20% | 20% |
| Membership Income | – | R$50,000 |
| **Total Profit** | **R$330K** | **R$445K** |

➡ Profit increase: **+R$115K/month** through ML-powered pricing insights.

---

## 🔗 Relevant Links

- Access Dataset and Model here: https://drive.google.com/drive/folders/1ujXQRfHrQEJQd03RCn6wDSkgamlikWOH?usp=sharing
- Access Tableau here: https://public.tableau.com/app/profile/uriel.siboro/viz/BrazilianE-CommerceFINAL/Dashboard2?publish=yes

---

## 🛠️ Tech Stack  

| Area | Tools |
|------|-------|
| Data Wrangling | Python, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly, Folium |
| ML Modeling | Scikit-learn, LightGBM, XGBoost |
| Text & Sentiment | WordCloud, NLP Preprocessing |
| Deployment | Streamlit (for price dashboard) |
| Project Management | GitHub, Jupyter Notebook |

---

