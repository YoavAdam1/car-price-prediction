import os
import kagglehub
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# הגדרות מפתח ותצוגה ב-Streamlit
st.set_page_config(page_title="תחזית מחיר רכב (Linear Regression)", layout="wide")

st.title("🚗 מחשבון תחזית מחיר רכב לפי קילומטראז'")
st.write("אפליקציה זו משתמשת בדגם רגרסיה ליניארית כדי לחזות מחיר רכב בהתבסס על הקילומטראז' שלו (km driven).")

# הגדרת פרמטרים
KAGGLE_DATASET = "nehalbirla/vehicle-dataset-from-cardekho"
FEATURE_COLUMN = "km_driven"
TARGET_COLUMN = "selling_price"

# טעינת נתונים עם Cache כדי לא להוריד מחדש בכל רענון
@st.cache_data
def load_data():
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    csv_path = os.path.join(path, "CAR DETAILS FROM CAR DEKHO.csv")
    df = pd.read_csv(csv_path)
    return df

with st.spinner('טוען נתונים מ-Kaggle...'):
    data = load_data()

# סרגל צד (Sidebar) למידע כללי וסטטיסטיקות
st.sidebar.header("📊 מידע על הנתונים")
st.sidebar.write(f"**סה\"כ רשומות:** {len(data)}")
st.sidebar.write(f"**חוסרים ב-{FEATURE_COLUMN}:** {data[FEATURE_COLUMN].isnull().sum()}")
st.sidebar.write(f"**חוסרים ב-{TARGET_COLUMN}:** {data[TARGET_COLUMN].isnull().sum()}")

# אימון הדגם
X = data[FEATURE_COLUMN].to_numpy()
y = data[TARGET_COLUMN].to_numpy()

X_reshaped = X.reshape(-1, 1)

model = LinearRegression()
model.fit(X_reshaped, y)

w = model.coef_[0]
b = model.intercept_

# חישוב שגיאות (Baseline vs Model Loss)
baseline_prediction = np.mean(y)
baseline_loss = np.mean(np.abs(y - baseline_prediction))

y_hat = model.predict(X_reshaped)
model_loss = np.mean(np.abs(y - y_hat))

# חלק 1: חיזוי מחיר בזמן אמת לפי קלט משתמש
st.subheader("🔮 בצע חיזוי מחיר רכב")

col1, col2 = st.columns([1, 1])

with col1:
    user_km = st.number_input(
        "הכנס מספר קילומטרים (km driven):", 
        min_value=0, 
        max_value=1000000, 
        value=50000, 
        step=1000
    )

    if st.button("חשב תחזית מחיר"):
        predicted_price = model.predict(np.array([[user_km]]))[0]
        st.success(f"💰 **המחיר החזוי:** {predicted_price:,.2f}")

with col2:
    st.markdown("### 📈 פרמטרים של המודל")
    st.write(f"**שיפוע (Slope / w):** `{w:.4f}`")
    st.write(f"**חיתוך (Intercept / b):** `{b:,.2f}`")
    st.write(f"**שגיאת בסיס (Baseline MAE):** `{baseline_loss:,.2f}`")
    st.write(f"**שגיאת מודל (Model MAE):** `{model_loss:,.2f}`")

# חלק 2: הצגת הנתונים והגרף
st.divider()

tab1, tab2, tab3 = st.tabs(["📉 גרף הרגרסיה", "📋 טבלת הנתונים (5 שורות ראשונות)", "📊 סטטיסטיקה תיאורית"])

with tab1:
    st.write("ויזואליזציה של נתוני האמת מול קו הרגרסיה הליניארית:")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(X, y, alpha=0.3, color="blue", label="נתונים אמיתיים")
    
    # יצירת קו רגרסיה
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    ax.plot(X_line, y_line, color="red", linewidth=2, label="קו רגרסיה (תחזית)")
    
    ax.set_xlabel("קילומטראז' (km_driven)")
    ax.set_ylabel("מחיר מכירה (selling_price)")
    ax.legend()
    st.pyplot(fig)

with tab2:
    st.dataframe(data.head())

with tab3:
    st.dataframe(data[[FEATURE_COLUMN, TARGET_COLUMN]].describe())
