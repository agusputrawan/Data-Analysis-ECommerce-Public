import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
sns.set(style='dark')

#Dataframe untuk Distribusi Lokasi Pelanggan dan Penjual
def create_location_distribution_df(df):
    customer_state_df = df.groupby("customer_state")["customer_id"].nunique().reset_index()
    customer_state_df.rename(columns={"customer_id": "customer_count"}, inplace=True)
    seller_state_df = df.groupby("seller_state")["seller_id"].nunique().reset_index()
    seller_state_df.rename(columns={"seller_id": "seller_count"}, inplace=True)
    state_distribution_df = pd.merge(customer_state_df, seller_state_df, 
                                     left_on="customer_state", right_on="seller_state", how="outer")
    state_distribution_df.rename(columns={"customer_state": "state"}, inplace=True)
    state_distribution_df.drop(columns=["seller_state"], inplace=True)
    customer_city_df = df.groupby("customer_city")["customer_id"].nunique().reset_index()
    customer_city_df.rename(columns={"customer_id": "customer_count"}, inplace=True)
    customer_city_df = customer_city_df.sort_values(by="customer_count", ascending=False).head(10)
    seller_city_df = df.groupby("seller_city")["seller_id"].nunique().reset_index()
    seller_city_df.rename(columns={"seller_id": "seller_count"}, inplace=True)
    seller_city_df = seller_city_df.sort_values(by="seller_count", ascending=False).head(10)

    return state_distribution_df, customer_city_df, seller_city_df

# Dataframe untuk Waktu Pengiriman
def create_delivery_time_df(df):
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
    df['delivery_time'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    delivery_time_df = df[['order_id', 'customer_state', 'customer_city', 'delivery_time']].dropna()
    mean_delivery_time = delivery_time_df['delivery_time'].mean()
    return delivery_time_df, mean_delivery_time

# Dataframe untuk Pengaruh Ulasan Terhadap Penjualan
def create_review_sales_df(df):
    review_sales_df = df.groupby('review_score', as_index=False).agg({
        'order_id': 'count'
    })
    review_sales_df.rename(columns={'order_id': 'order_count'}, inplace=True)
    return review_sales_df

# Dataframe Produk Paling Banyak Menyumbang Pendapatan
def create_category_revenue_df(df):
    category_revenue_df = df.groupby('product_category_name_english', as_index=False).agg({
        'price': 'sum'
    })
    category_revenue_df.rename(columns={'price': 'total_revenue'}, inplace=True)
    category_revenue_df = category_revenue_df.sort_values(by='total_revenue', ascending=False)
    return category_revenue_df

# Dataframe 10 Produk Dengan Pendapatan Tertinggi, Metode Pembayaran Apa Yang Paling Sering Digunakan
def create_top_products_payment_df(df):
    df['total_revenue'] = df['price'] * df['order_item_id']
    top_10_products = df.groupby('product_category_name_english', as_index=False)['total_revenue'].sum()
    top_10_products = top_10_products.sort_values(by='total_revenue', ascending=False).head(10)
    top_products_df = df[df['product_category_name_english'].isin(top_10_products['product_category_name_english'])]
    payment_distribution_df = top_products_df.groupby(['product_category_name_english', 'payment_type'])['order_id'].count().unstack(fill_value=0)
    return top_10_products, payment_distribution_df

# Dataframe untuk RFM Analysis
def create_rfm_df(df):
    rfm_df = df[['customer_id', 'order_purchase_timestamp', 'order_id', 'price']].copy()
    rfm_df['order_purchase_timestamp'] = pd.to_datetime(rfm_df['order_purchase_timestamp'])
    now = rfm_df['order_purchase_timestamp'].max()
    rfm_df['recency'] = (now - rfm_df['order_purchase_timestamp']).dt.days
    rfm_df['frequency'] = rfm_df.groupby('customer_id')['order_id'].transform('count')
    rfm_df['monetary'] = rfm_df.groupby('customer_id')['price'].transform('sum')
    rfm_analysis = rfm_df.groupby('customer_id').agg({
        'recency': 'min',    
        'frequency': 'max',  
        'monetary': 'max'   
    }).reset_index()
    return rfm_analysis

# Load dataset
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "main_data.csv")
    df = pd.read_csv(file_path)
    return df
all_df = load_data()

# Komponen Side Bar
all_df['order_purchase_timestamp'] = pd.to_datetime(all_df['order_purchase_timestamp'], errors='coerce')
all_df = all_df.sort_values(by="order_purchase_timestamp").reset_index(drop=True)

# Ambil rentang tanggal dari dataset
min_date = all_df["order_purchase_timestamp"].min().date()
max_date = all_df["order_purchase_timestamp"].max().date()

# Membuat Sidebar
with st.sidebar:
    # Menambahkan logo perusahaan
    # st.image("logo.png", width=270)
    # Menambahkan filter tangal
    st.markdown(
        "<h4 style='text-align: center;'>📅 Rentang Waktu Transaksi</h4>",
        unsafe_allow_html=True
    )
    start_date, end_date = st.date_input(
        label="",
        min_value=min_date,
         max_value=max_date,
         value=[min_date, max_date]
)
filtered_df = all_df[
    (all_df["order_purchase_timestamp"].dt.date >= start_date) & 
    (all_df["order_purchase_timestamp"].dt.date <= end_date)
]

# Menyiapkan berbagai Dataframe
state_distribution_df, customer_city_df, seller_city_df = create_location_distribution_df(all_df)
delivery_time_df, mean_delivery_time = create_delivery_time_df(all_df)
review_sales_df = create_review_sales_df(all_df)
category_revenue_df = create_category_revenue_df(all_df)
top_10_products_df, payment_distribution_df = create_top_products_payment_df(all_df)
rfm_analysis_df = create_rfm_df(all_df)


# Membuat Dashboard
st.markdown(
    "<h1 style='text-align: center;'> Dashboard E-Commerce Analysis</h1>", 
    unsafe_allow_html=True
)

# Chart Distribusi Lokasi Pelanggan dan Penjual
st.markdown(
    "<h3 style='text-align: center;'>Distribusi Lokasi Pelanggan dan Penjual</h3>", 
    unsafe_allow_html=True
)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Customer distribution by state
sns.countplot(x='customer_state', data=all_df, ax=axes[0])
axes[0].set_title('Distribusi Lokasi Pelanggan menurut Negara Bagian')
axes[0].tick_params(axis='x', rotation=45, labelsize=7)

# Customer distribution by city (Top 10)
sns.countplot(x='customer_city', data=all_df.head(10), ax=axes[1])
axes[1].set_title('Distribusi Lokasi Pelanggan Berdasarkan Kota (10 Teratas)')
axes[1].tick_params(axis='x', rotation=45, labelsize=8)

# Seller distribution by state
sns.countplot(x='seller_state', data=all_df, ax=axes[2])
axes[2].set_title('Distribusi Lokasi Penjual menurut Negara Bagian')
axes[2].tick_params(axis='x', rotation=45, labelsize=8)

plt.tight_layout()

# Tampilkan chart di Streamlit
st.pyplot(fig)

# Chart Waktu Pengiriman
st.markdown(
    "<h3 style='text-align: center;'>Analisis Waktu Pengiriman</h3>", 
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center; font-size: 18px;'> Rata-rata Waktu Pengiriman</h4>", 
    unsafe_allow_html=True
)

st.write(f"📦 Rata-rata waktu pengiriman: **{mean_delivery_time:.2f} hari**")

st.markdown(
    "<h4 style='text-align: center; font-size: 18px;'> Distribusi Waktu Pengiriman</h4>", 
    unsafe_allow_html=True
)

fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(delivery_time_df['delivery_time'], kde=True, ax=ax)
ax.set_title("Distribusi Waktu Pengiriman")
ax.set_xlabel("Waktu Pengiriman (Hari)")
ax.set_ylabel("Frekuensi")
ax.axvline(mean_delivery_time, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_delivery_time:.2f} days')
ax.legend()

st.pyplot(fig)

# Chart Pengaruh Ulasan Terhadap Penjualan
st.markdown(
    "<h3 style='text-align: center;'>Pengaruh Ulasan terhadap Penjualan</h3>", 
    unsafe_allow_html=True
) 
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=review_sales_df['review_score'], y=review_sales_df['order_count'], ax=ax)
ax.set_title("Jumlah Pesanan berdasarkan Skor Ulasan")
ax.set_xlabel("Skor Ulasan")
ax.set_ylabel("Jumlah Pesanan")

st.pyplot(fig)

# Chart Produk Paling Banyak Menyumbang Pendapatan
st.markdown(
    "<h3 style='text-align: center;'>Kategori Produk dengan Pendapatan Tertinggi</h3>", 
    unsafe_allow_html=True
)

fig, ax = plt.subplots(figsize=(12, 6))
top_10_categories = category_revenue_df.head(10)
sns.barplot(x=top_10_categories['product_category_name_english'], 
            y=top_10_categories['total_revenue'], ax=ax)

ax.set_title("Top 10 Kategori Produk berdasarkan Pendapatan")
ax.set_xlabel("Kategori Produk")
ax.set_ylabel("Total Pendapatan")
plt.xticks(rotation=45, ha='right')

st.pyplot(fig)

# Chart 10 Produk Dengan Pendapatan Tertinggi, Metode Pembayaran Apa Yang Paling Sering Digunakan
st.markdown(
    "<h3 style='text-align: center;'>Top 10 Produk dengan Pendapatan Tertinggi & Metode Pembayaran</h3>", 
    unsafe_allow_html=True
)
fig, ax = plt.subplots(figsize=(12, 6))
payment_distribution_df.plot(kind='bar', stacked=True, ax=ax)

ax.set_title("Distribusi Metode Pembayaran untuk 10 Produk Teratas")
ax.set_xlabel("Produk Kategori")
ax.set_ylabel("Jumlah Pesanan")
plt.xticks(rotation=45, ha='right')
plt.legend(title='Metode Pembayaran')

st.pyplot(fig)


# Chart RFM Analysis
st.markdown(
    "<h3 style='text-align: center;'>📊 RFM Analysis</h3>", 
    unsafe_allow_html=True
)

# Visualisasi Distribusi RFM
st.markdown(
    "<h4 style='text-align: center; font-size: 18px;'> Distribusi RFM Metrics</h4>", 
    unsafe_allow_html=True
)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Recency Distribution
sns.histplot(rfm_analysis_df['recency'], kde=True, color='skyblue', ax=axes[0])
axes[0].set_title('Recency Distribution')
axes[0].set_xlabel('Recency (Days)')
axes[0].set_ylabel('Frequency')

# Frequency Distribution
sns.histplot(rfm_analysis_df['frequency'], kde=True, color='lightgreen', bins=50, ax=axes[1])
axes[1].set_xlim(0, np.percentile(rfm_analysis_df['frequency'], 99))  # Memotong outlier
axes[1].set_title('Frequency Distribution')
axes[1].set_xlabel('Frequency (Orders)')
axes[1].set_ylabel('Frequency')

# Monetary Distribution (Menggunakan Log untuk menangani skewed data)
sns.histplot(np.log1p(rfm_analysis_df['monetary']), kde=True, color='lightcoral', ax=axes[2])
axes[2].set_title('Monetary Distribution')
axes[2].set_xlabel('Monetary Value (Price)')
axes[2].set_ylabel('Frequency')

plt.tight_layout()
st.pyplot(fig)

# Kesimpulan
st.header("Kesimpulan")
st.write("""
##  **Strategi Bisnis Berdasarkan RFM Analysis**  
| **Aspek**      | **Insight** | **Strategi** |
|---------------|-----------|------------|
| **Recency** (Kapan terakhir belanja?) | Banyak pelanggan lama yang belum kembali berbelanja | Strategi reaktivasi pelanggan lama: Email promo, diskon eksklusif, program loyalitas |
| **Frequency** (Seberapa sering pelanggan belanja?) | Sebagian besar pelanggan hanya belanja 1 kali | Strategi retensi pelanggan: Program repeat purchase, promo khusus pelanggan baru |
| **Monetary** (Seberapa besar pengeluaran pelanggan?) | Ada dua segmen pelanggan (pembeli kecil vs pembeli besar) | Segmentasi & personalisasi promo sesuai nilai pelanggan |

---

## ✨ **Rekomendasi Bisnis untuk Pertumbuhan** 🚀  
✅ **Reaktivasi pelanggan lama** → Kirim email reminder & diskon eksklusif.  
✅ **Dorong pembelian berulang** → Gunakan promo spesial untuk pembelian kedua.  
✅ **Optimalkan kategori produk terlaris** → Fokus marketing & stok pada kategori Health & Beauty.  
✅ **Tingkatkan kecepatan pengiriman** → Sediakan opsi ekspres & optimalkan logistik.  
✅ **Gunakan metode pembayaran populer** → Dorong lebih banyak transaksi dengan promo khusus kartu kredit & invoice. """)                  



st.caption(f"Copyright © 2024 All Rights Reserved [Putu Agus Putrawan](https://www.linkedin.com/in/putu-agus-putrawan/)")


