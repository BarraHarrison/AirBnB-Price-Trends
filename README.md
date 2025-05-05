# 🏙️ Airbnb Price Trends - NYC

This project explores Airbnb listings in New York City using **Exploratory Data Analysis (EDA)** and presents the findings in an interactive **Streamlit dashboard**. My analysis examines how listing price varies by room type, availability, location, and proximity to famous NYC landmarks.

---

## 📂 Dataset

* **Source**: [Inside Airbnb NYC Dataset (2019)](http://insideairbnb.com/get-the-data.html)
* **File Used**: `AB_NYC_2019.csv`
* **Features include**:

  * Room type
  * Neighborhood group
  * Price
  * Availability
  * Coordinates (latitude & longitude)
  * Host activity

---

## 🧪 EDA Highlights

### 🔍 Key Insights:

* **Price vs. Location**: Interactive scatterplots reveal that central Manhattan and Brooklyn tend to have higher prices.
* **Room Type Distribution**: Private rooms are the most common type, followed by entire homes/apartments.
* **Availability Patterns**: Listings are either rarely available or available almost year-round (bimodal distribution).
* **Host Activity**: Sparse hosting patterns with a minority of listings highly active.
* **Distance to Landmarks**: Prices generally decrease as distance increases from popular locations like Times Square, Central Park, and DUMBO.

---

## 🧭 Streamlit Dashboard Features

### 🔧 Sidebar Filters:

* Filter listings by **room type** and **price range**

### 📊 Visualizations:

* **Map of Listings**: Visualize filtered listings on a live map
* **Price Distribution**: Color-coded scatterplot of prices across NYC
* **Room Type Bar Chart**: Frequency of each room type
* **Availability Histogram**: Availability rates grouped by room type
* **Summary Table**: Descriptive statistics of the filtered dataset
* **Landmark Analysis**: Price vs. distance scatterplots for Times Square, Central Park, and DUMBO

---

## 🚀 How to Run the Dashboard

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2. **Run the app locally**:

```bash
streamlit run airbnb_dashboard.py
```

3. **Access the app**:
   Open your browser and go to: `http://localhost:8501`

---

## 📁 Project Structure

```
AirBnB-Price-Trends/
├── AB_NYC_2019.csv
├── airbnb_dashboard.py
├── requirements.txt
└── README.md
```

---

## 📌 Future Improvements

* Integrate time-based trends (e.g., seasonality by review date)
* Add clustering visualization by borough or neighborhood
* Deploy dashboard on Streamlit Cloud for public access

---

## 📬 Contact

EDA by [Barra Harrison](https://github.com/BarraHarrison)
For feedback or collaboration, feel free to connect on [LinkedIn](https://www.linkedin.com/in/barraharrison20091997/)
