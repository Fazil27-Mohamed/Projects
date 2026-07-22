import streamlit as st
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
st.set_page_config(
    page_title="AI Product Recommendation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_data():
    df = joblib.load("products.pkl")
    X_scaled = joblib.load("X_scaled.pkl")
    return df, X_scaled

df, X_scaled = load_data()

st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#0f172a,#111827,#1e3a8a);
color:white;
}

.main-title{
font-size:45px;
font-weight:bold;
text-align:center;
background:linear-gradient(90deg,#00F5A0,#00D9F5,#A100FF);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
margin-bottom:5px;
}

.subtitle{
text-align:center;
font-size:18px;
color:#d1d5db;
margin-bottom:30px;
}

.block{
background:rgba(255,255,255,.08);
padding:20px;
border-radius:20px;
backdrop-filter:blur(15px);
box-shadow:0px 5px 20px rgba(0,0,0,.35);
margin-bottom:20px;
}

.metric{
background:rgba(255,255,255,.08);
padding:18px;
border-radius:18px;
text-align:center;
}

.product-card{
background:rgba(255,255,255,.08);
padding:20px;
border-radius:20px;
margin-top:10px;
margin-bottom:15px;
transition:0.3s;
}

.product-card:hover{
transform:scale(1.02);
box-shadow:0px 0px 20px cyan;
}

div.stButton > button{
width:100%;
height:55px;
font-size:18px;
font-weight:bold;
border-radius:15px;
background:linear-gradient(90deg,#00DBDE,#FC00FF);
color:white;
border:none;
}

div.stButton > button:hover{
background:white;
color:black;
}

</style>
""", unsafe_allow_html=True)


st.markdown(
    "<div class='main-title'>️ AI Product Recommendation System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Find Similar Products using Machine Learning</div>",
    unsafe_allow_html=True
)

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.metric("Products", len(df))

with c2:
    st.metric("Categories", df["category"].nunique())

with c3:
    st.metric("Brands", df["brand"].nunique())

with c4:
    st.metric("Average Rating", round(df["rating"].mean(),2))

st.divider()


st.sidebar.title("⚙️ Filters")

category = st.sidebar.selectbox(
    "Category",
    ["All"] + sorted(df["category"].dropna().unique())
)

brand = st.sidebar.selectbox(
    "Brand",
    ["All"] + sorted(df["brand"].dropna().unique())
)

products = sorted(df["product_name"].unique())

selected_product = st.selectbox(
    "🔍 Select Product",
    products
)

def recommend(product_name, n=10):

    selected = df[df["product_name"] == product_name].iloc[0]

    filtered = df[df["category"] == selected["category"]]

    brand_filtered = filtered[
        filtered["brand"] == selected["brand"]
    ]

    if len(brand_filtered) >= 10:
        filtered = brand_filtered

    price = selected["final_price"]

    price_filtered = filtered[
        (filtered["final_price"] >= price * 0.8)
        &
        (filtered["final_price"] <= price * 1.2)
    ]

    if len(price_filtered) >= 10:
        filtered = price_filtered

    rating = selected["rating"]

    rating_filtered = filtered[
        (filtered["rating"] >= rating - 1)
        &
        (filtered["rating"] <= rating + 1)
    ]

    if len(rating_filtered) >= 10:
        filtered = rating_filtered

    idx = filtered.index

    X_filtered = X_scaled[idx]

    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=min(n + 1, len(filtered))
    )

    model.fit(X_filtered)

    query_index = list(idx).index(selected.name)

    distances, indices = model.kneighbors(
        X_filtered[query_index].reshape(1, -1)
    )

    result = filtered.iloc[indices[0][1:]].copy()

    result["Similarity"] = (
        (1 - distances[0][1:]) * 100
    ).round(2)

    return result[
        [
            "product_name",
            "category",
            "brand",
            "final_price",
            "rating",
            "Similarity"
        ]
    ]


if st.button("🚀 Get Recommendations"):

    with st.spinner("Finding Similar Products..."):

        recommendations = recommend(selected_product)

    st.success(
        f"{len(recommendations)} Recommendations Found"
    )

    st.divider()

    for _, row in recommendations.iterrows():

        with st.container():

            st.markdown(
                "<div class='product-card'>",
                unsafe_allow_html=True
            )

            c1, c2 = st.columns([4,1])

            with c1:

                st.subheader(row["product_name"])

                st.write(
                    f"**Category :** {row['category']}"
                )

                st.write(
                    f"**Brand :** {row['brand']}"
                )

                st.write(
                    f"**Price : ₹{row['final_price']:,}"
                )

                st.write(
                    f"**Rating : ⭐ {row['rating']}"
                )

            with c2:

                st.metric(
                    "Similarity",
                    f"{row['Similarity']}%"
                )

            st.progress(
                min(row["Similarity"]/100,1.0)
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

st.divider()

st.header("📊 Product Analytics Dashboard")

tab1, tab2, tab3 = st.tabs(
    ["📦 Categories", "⭐ Ratings", "💰 Prices"]
)

with tab1:

    category_df = (
        df["category"]
        .value_counts()
        .reset_index()
    )

    category_df.columns = [
        "Category",
        "Products"
    ]

    fig = px.bar(
        category_df,
        x="Category",
        y="Products",
        color="Products",
        text="Products",
        template="plotly_dark",
        title="Products in Each Category"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with tab2:

    fig = px.histogram(
        df,
        x="rating",
        nbins=20,
        template="plotly_dark",
        title="Product Rating Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with tab3:

    fig = px.box(
        df,
        y="final_price",
        color="category",
        template="plotly_dark",
        title="Price Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

st.header("🔥 Top 10 Brands")

brand_df = (
    df["brand"]
    .value_counts()
    .head(10)
    .reset_index()
)

brand_df.columns = [
    "Brand",
    "Products"
]

fig = px.bar(
    brand_df,
    x="Brand",
    y="Products",
    color="Products",
    text="Products",
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

with st.expander("📄 View Dataset"):

    st.dataframe(
        df,
        use_container_width=True,
        height=500
    )

st.divider()

st.markdown(
"""
<div style='text-align:center;padding:25px;'>

<h2>🛍️ AI Product Recommendation System</h2>

</div>
""",
unsafe_allow_html=True
)
