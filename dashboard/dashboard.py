# Sentiment Monitoring Dashboard - Streamlit Frontend
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime
import time
import io


# SESSION MANAGEMENT
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None
if "results_df" not in st.session_state:
    st.session_state["results_df"] = pd.DataFrame(
        columns=["text", "sentiment", "aspect", "confidence", "timestamp"]
    )
if "live_running" not in st.session_state:
    st.session_state["live_running"] = False

API_BASE = "http://127.0.0.1:5000"


# LOGIN SIDEBAR
if not st.session_state["logged_in"]:
    with st.sidebar:
        st.title("User Login")
        username_input = st.text_input("Username")
        password_input = st.text_input("Password", type="password")
        if st.button("Login"):
            try:
                response = requests.post(
                    f"{API_BASE}/login",
                    json={"username": username_input, "password": password_input},
                    timeout=10
                )
                if response.status_code == 200:
                    user_data = response.json()
                    st.session_state["logged_in"] = True
                    st.session_state["user_id"] = user_data["user_id"]
                    st.session_state["username"] = username_input
                    st.success(f"Logged in as {username_input}")
                else:
                    st.error("Invalid credentials")
            except requests.exceptions.RequestException:
                st.error("Failed to connect to the login API.")


# DASHBOARD
if st.session_state["logged_in"]:
    st.title(f"Sentiment Monitoring Dashboard - {st.session_state['username']}")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Single Message", "Batch Upload", "Filtered Reports", "Live Mode"]
    )


    # Single Message Analysis
    with tab1:
        st.subheader("Analyze a Single Message (Real-Time)")
        with st.form("single_message_form"):
            user_input = st.text_area("Enter customer feedback")
            submitted = st.form_submit_button("Analyze Sentiment")
            if submitted and user_input.strip():
                try:
                    response = requests.post(
                        f"{API_BASE}/predict-sentiment",
                        json={"text": user_input, "user_id": st.session_state["user_id"]},
                        timeout=10
                    )
                    if response.status_code == 200:
                        result = response.json()
                        result["text"] = user_input
                        result["timestamp"] = datetime.now()
                        st.session_state["results_df"] = pd.concat(
                            [st.session_state["results_df"], pd.DataFrame([result])],
                            ignore_index=True
                        )
                        st.success("Analysis Complete")
                        st.write("**Sentiment:**", result["sentiment"])
                        st.write("**Confidence:**", f"{result['confidence']:.2f}")
                        st.write("**Aspect:**", result["aspect"])
                    else:
                        st.error("Error analyzing text.")
                except requests.exceptions.RequestException:
                    st.error("Failed to connect to the analysis API.")


    # Batch Upload
    def analyze_batch(df_upload):
        results = []
        progress_bar = st.progress(0)
        for i, text in enumerate(df_upload["text"]):
            try:
                response = requests.post(
                    f"{API_BASE}/predict-sentiment",
                    json={"text": str(text), "user_id": st.session_state["user_id"]},
                    timeout=10
                )
                if response.status_code == 200:
                    res = response.json()
                    res["text"] = text
                    res["timestamp"] = datetime.now()
                    results.append(res)
                else:
                    results.append({
                        "text": text, "sentiment": None, "aspect": None,
                        "confidence": None, "timestamp": datetime.now()
                    })
            except Exception as e:
                st.warning(f"Skipping row {i} due to error: {e}")
                results.append({
                    "text": text, "sentiment": None, "aspect": None,
                    "confidence": None, "timestamp": datetime.now()
                })
            progress_bar.progress((i + 1) / len(df_upload))
            time.sleep(0.1)
        return pd.DataFrame(results)

    with tab2:
        st.subheader("Batch Upload Feedback")
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
        if uploaded_file:
            df_upload = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            if "text" not in df_upload.columns:
                st.error("File must contain a 'text' column.")
            else:
                if st.button("Analyze Uploaded File"):
                    batch_results = analyze_batch(df_upload)
                    st.session_state["results_df"] = pd.concat([st.session_state["results_df"], batch_results], ignore_index=True)
                    st.dataframe(batch_results)
                    csv = batch_results.to_csv(index=False).encode("utf-8")
                    st.download_button("Download Batch Results", csv, "batch_results.csv", "text/csv")


    # Filtered Reports
    with tab3:
        st.subheader("Generate Reports with Filters")
        df = st.session_state["results_df"]
        if not df.empty:
            col1, col2, col3 = st.columns(3)
            start_date = col1.date_input("Start Date", value=df["timestamp"].min().date())
            end_date = col2.date_input("End Date", value=df["timestamp"].max().date())
            sentiment_filter = col3.multiselect("Sentiment", options=df["sentiment"].dropna().unique(), default=df["sentiment"].dropna().unique())
            aspect_filter = st.multiselect("Aspect", options=df["aspect"].dropna().unique(), default=df["aspect"].dropna().unique())
            
            filtered_df = df[
                (df["timestamp"].dt.date >= start_date) &
                (df["timestamp"].dt.date <= end_date) &
                (df["sentiment"].isin(sentiment_filter)) &
                (df["aspect"].isin(aspect_filter))
            ]
            st.dataframe(filtered_df)
            if not filtered_df.empty:
                st.download_button(
                    "Download Filtered Report",
                    filtered_df.to_csv(index=False).encode("utf-8"),
                    "filtered_report.csv",
                    "text/csv"
                )
        else:
            st.info("No data available yet.")


    # Live Mode
    with tab4:
        st.subheader("Live Feedback Simulation (CSV Input)")
        uploaded_live_file = st.file_uploader("Upload CSV for live feed", type=["csv"], key="live_csv")
        if uploaded_live_file:
            live_df = pd.read_csv(uploaded_live_file)
            if "text" not in live_df.columns:
                st.error("CSV must contain a 'text' column for feedback.")
            else:
                if st.button("Start Live Stream"):
                    st.session_state["live_running"] = True
                if st.button("Stop Live Stream"):
                    st.session_state["live_running"] = False

                if st.session_state["live_running"]:
                    st.info("Live streaming active... new feedback from CSV every 2 seconds")
                    for i, row in live_df.iterrows():
                        if not st.session_state["live_running"]:
                            break
                        text = str(row["text"])
                        try:
                            response = requests.post(
                                f"{API_BASE}/predict-sentiment",
                                json={"text": text, "user_id": st.session_state["user_id"]},
                                timeout=10
                            )
                            if response.status_code == 200:
                                result = response.json()
                                result["text"] = text
                                result["timestamp"] = datetime.now()
                                st.session_state["results_df"] = pd.concat(
                                    [st.session_state["results_df"], pd.DataFrame([result])],
                                    ignore_index=True
                                )
                                st.success(f"New feedback analyzed: {text}")
                        except requests.exceptions.RequestException:
                            st.error(f"Failed to analyze: {text}")
                        time.sleep(2)


    # KPIs & Charts
    st.subheader("Analytics & Insights")
    data = st.session_state["results_df"]
    if not data.empty:
        total_feedback = len(data)
        negative_rate = data["sentiment"].value_counts().get("negative", 0) / total_feedback * 100
        top_aspect = data["aspect"].value_counts().idxmax()
        avg_conf = data["confidence"].mean()

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Feedback", total_feedback)
        kpi2.metric("Negative Rate", f"{negative_rate:.1f}%")
        kpi3.metric("Top Topic", top_aspect)
        kpi4.metric("Avg Confidence", f"{avg_conf:.2f}")

        # Sentiment Pie Chart
        st.subheader("Sentiment Distribution")
        pie_df = data["sentiment"].value_counts().reset_index()
        pie_df.columns = ["sentiment", "count"]
        fig1 = px.pie(pie_df, names="sentiment", values="count")
        st.plotly_chart(fig1, use_container_width=True)
        # Download Sentiment Pie
        buf1 = io.BytesIO()
        fig1.write_image(buf1, format="png")
        st.download_button("Download Sentiment Pie", buf1, "sentiment_pie.png", "image/png")

        # Aspect Bar Chart
        st.subheader("Aspect Distribution")
        aspect_df = data["aspect"].value_counts().reset_index()
        aspect_df.columns = ["aspect", "count"]
        fig2 = px.bar(aspect_df, x="aspect", y="count")
        st.plotly_chart(fig2, use_container_width=True)
        # Download Aspect Bar
        buf2 = io.BytesIO()
        fig2.write_image(buf2, format="png")
        st.download_button("Download Aspect Bar", buf2, "aspect_bar.png", "image/png")

        # Recent Negative Feedback
        st.subheader("Recent Negative Feedback")
        negative = data[data["sentiment"] == "negative"].tail(5)
        for _, row in negative.iterrows():
            st.error(f"⚠ Message: {row['text']} | Aspect: {row['aspect']} | Confidence: {row['confidence']:.2f}")

        # Automated Insights
        st.subheader("Automated Insights")
        insights = []
        for aspect in data["aspect"].dropna().unique():
            aspect_df = data[data["aspect"] == aspect]
            neg_pct = aspect_df["sentiment"].value_counts().get("negative", 0) / len(aspect_df) * 100
            insights.append(f"- {aspect}: {neg_pct:.1f}% negative feedback")
        st.markdown("\n".join(insights))

        # Download Full Analytics CSV
        st.subheader("Download Full Analytics Data")
        full_csv = data.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Full Analytics CSV",
            full_csv,
            "full_analytics.csv",
            "text/csv"
        )

else:
    st.title("Please login to access the dashboard.")