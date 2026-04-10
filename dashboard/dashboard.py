# Sentiment Monitoring Dashboard - Streamlit Frontend

import sys
import os
import io
import time
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ENVIRONMENT SETUP
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

# IMPORTS
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Local models & DB
from app.models.finbert_model import predict_sentiment
from app.models.absa_model import analyze_aspects
from app.utils.db import save_result, get_all_results 

# CONFIG
API_BASE = st.secrets.get("API_BASE") or os.getenv("API_BASE")


# HELPER FUNCTIONS

def analyze_batch(df_upload):
    results = []
    total = len(df_upload)

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, text in enumerate(df_upload["text"]):
        status_text.text(f"Processing {i+1} of {total}...")

        try:
            res = call_predict_sentiment(str(text))
            res["timestamp"] = datetime.now()
            results.append(res)

            try:
                save_result(
                    text=res.get("text", str(text)),
                    aspect=res.get("aspect", "Unknown"),
                    sentiment=res.get("sentiment", "neutral"),
                    confidence=float(res.get("confidence", 0.0)),
                    timestamp=res["timestamp"]
                )
            except Exception as e:
                st.warning(f"Could not save row {i} to DB: {e}")

        except Exception as e:
            st.warning(f"Error occurred while processing row {i}: {e}")
            results.append({
                "text": str(text),
                "sentiment": None,
                "aspect": None,
                "confidence": None,
                "timestamp": datetime.now()
            })

        progress_bar.progress(int((i + 1) / total * 100))

    st.session_state["results_df"] = pd.concat(
        [st.session_state.get("results_df", pd.DataFrame()), pd.DataFrame(results)],
        ignore_index=True
    )

    status_text.text("✅ Processing complete!")

    return pd.DataFrame(results)

def call_predict_sentiment(text, user_id=None):
    try:
        if API_BASE:
            response = requests.post(
                f"{API_BASE}/predict-sentiment",
                json={"text": text, "user_id": user_id},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        else:
            sentiment_result = predict_sentiment(text)
            aspect_result = analyze_aspects(text)

            aspects = aspect_result.get("aspects", [])
            aspect = aspects[0].get("aspect", "general") if aspects else "general"

            return {
                "text": text,
                "sentiment": sentiment_result.get("sentiment", "neutral"),
                "confidence": sentiment_result.get("confidence", 0.0),
                "aspect": aspect
            }

    except Exception as e:
        st.error(f"Inference failed: {e}")
        return {
            "text": text,
            "sentiment": "error",
            "confidence": 0.0,
            "aspect": "error"
        }


def generate_pdf_report(data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Sentiment Analysis Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    total = len(data)
    negative = data["sentiment"].value_counts().get("negative", 0)
    negative_rate = (negative / total * 100) if total > 0 else 0
    top_aspect = data["aspect"].value_counts().idxmax() if not data.empty else "N/A"

    elements.append(Paragraph(f"Total Feedback: {total}", styles["Normal"]))
    elements.append(Paragraph(f"Negative Rate: {negative_rate:.2f}%", styles["Normal"]))
    elements.append(Paragraph(f"Top Aspect: {top_aspect}", styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer


# SESSION STATE

if "results_df" not in st.session_state:
    st.session_state["results_df"] = pd.DataFrame(
        columns=["text", "sentiment", "aspect", "confidence", "timestamp"]
    )


# SIDEBAR

st.sidebar.title("📊 Navigation")

menu = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "💬 Single Message",
        "📂 Batch Upload",
        "📡 Live Feedback Simulation",
        "📊 Reports",
        "📈 Analytics and Insights Dashboard" 
    ]
)

st.sidebar.markdown("### 💡 Feedback")
st.sidebar.markdown("👉 [Help us Improve this tool by leaving a review](https://forms.gle/M8W3Z2dnDiFZePFC7)")
st.sidebar.markdown("---")


# HOME

if menu == "🏠 Home":
    st.title("Sentiment Monitoring Dashboard")

    st.markdown("""
    ### Welcome 👋

    This system analyzes customer feedback using a fine-tuned FinBERT model.

    #### Features:
    - 💬 Sentiment analysis (Positive / Neutral / Negative)
    - 📂 Batch processing
    - 📡 Live feedback tracking
    - 📊 Reports 
    - 📈 Analytics and Insights Dashboard
    """)


# SINGLE MESSAGE

elif menu == "💬 Single Message":
    st.subheader("💬 Analyze a Single Message")

    with st.form("single_message_form", clear_on_submit=True):
        user_input = st.text_area("Enter customer feedback")
        submitted = st.form_submit_button("Analyze Sentiment")

    if submitted and user_input.strip():
        result = call_predict_sentiment(user_input)
        result["timestamp"] = datetime.now()

        st.session_state["results_df"] = pd.concat(
            [st.session_state["results_df"], pd.DataFrame([result])],
            ignore_index=True
        )

        try:
            save_result(**result)
        except:
            pass

        st.success("Analysis Complete")

        col1, col2, col3 = st.columns(3)
        col1.metric("Sentiment", result["sentiment"].capitalize())
        col2.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
        col3.metric("Aspect", result["aspect"].replace("_", " ").title())


# BATCH UPLOAD

elif menu == "📂 Batch Upload":
        st.subheader("📂 Batch Upload")

        st.markdown("### Upload a CSV or Excel file with a 'text' column to analyze multiple feedback entries at once")

        uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

        if uploaded_file:
            df_upload = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

            if "text" not in df_upload.columns:
                st.error("File must contain 'text' column.")
            else:
                if st.button("Analyze File", key="batch_analyze"):
                    
                    with st.spinner("Processing batch..."):
                        results_df = analyze_batch(df_upload)

                    st.dataframe(results_df)

                    st.download_button(
                        "Download Results",
                        results_df.to_csv(index=False).encode("utf-8"),
                        "results.csv",
                        key="download_batch_results"
                    )


# LIVE FEEDBACK SIMULATION

elif menu == "📡 Live Feedback Simulation":
    st.subheader("📡 Live Feedback Simulation")

    st.markdown("## Simulate real-time customer feedback and see live sentiment analysis results")

    if "live_results" not in st.session_state:
        st.session_state["live_results"] = pd.DataFrame(
            columns=["text", "sentiment", "aspect", "confidence", "timestamp"]
        )

    # --- Feedback submission form ---
    with st.form("live_feedback_form", clear_on_submit=True):
        new_feedback = st.text_input("Enter new feedback here")
        submitted = st.form_submit_button("Submit Feedback")

        if submitted and new_feedback.strip():
            try:
                result = call_predict_sentiment(new_feedback)

                result_data = {
                    "text": result.get("text", new_feedback),
                    "aspect": result.get("aspect", "Unknown"),
                    "sentiment": result.get("sentiment", "neutral"),
                    "confidence": float(result.get("confidence", 0.0)),
                    "timestamp": datetime.now()
                }

                # Save to live feed
                st.session_state["live_results"] = pd.concat(
                    [
                        st.session_state["live_results"],
                        pd.DataFrame([result_data])
                    ],
                    ignore_index=True
                )

                # Save to main dataset
                st.session_state["results_df"] = pd.concat(
                    [
                        st.session_state["results_df"],
                        pd.DataFrame([result_data])
                    ],
                    ignore_index=True
                )

                # Save to DB
                try:
                    save_result(
                        text=result_data["text"],
                        aspect=result_data["aspect"],
                        sentiment=result_data["sentiment"],
                        confidence=result_data["confidence"],
                    )
                except Exception as e:
                    st.warning(f"Could not save to DB: {e}")

                # ✅ Success message
                st.success("Feedback analyzed and saved!")

            except Exception as e:
                st.error(f"Failed to analyze feedback: {e}")

        # --- Display live feed ---
        st.subheader("Live Feed (Latest 10 Messages)")

        df = st.session_state["live_results"]

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            live_display_df = df.sort_values(
                by="timestamp", ascending=False
            ).head(10)

            # Format timestamp for readability
            live_display_df["timestamp"] = live_display_df["timestamp"].dt.strftime("%H:%M:%S")

            # Rename columns for nicer headers
            live_display_df = live_display_df.rename(columns={
                "timestamp": "Time",
                "text": "Feedback",
                "aspect": "Aspect",
                "sentiment": "Sentiment",
                "confidence": "Confidence"
            })

            st.dataframe(live_display_df, use_container_width=True)

        else:
            st.info("No feedback submitted yet.")


# REPORTS

elif menu == "📊 Reports":
    st.subheader("📊 Reports")

    st.markdown("##  Apply filters to generate custom reports")

    # Fetch all data from database
    results = get_all_results()
    db_df = pd.DataFrame(results)

    # Fetch session data
    session_df = st.session_state.get("results_df", pd.DataFrame())

    # Ensure same structure
    required_cols = ["text", "sentiment", "aspect", "confidence", "timestamp"]
    for col in required_cols:
        if col not in session_df.columns:
            session_df[col] = None

    # Combine safely
    if not db_df.empty and not session_df.empty:
        df = pd.concat([db_df, session_df], ignore_index=True)
    elif not db_df.empty:
        df = db_df
    else:
        df = session_df

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        col1, col2, col3 = st.columns(3)
        start_date = col1.date_input("Start Date", df["timestamp"].min().date())
        end_date = col2.date_input("End Date", df["timestamp"].max().date())

        sentiment_filter = col3.multiselect(
            "Sentiment",
            options=df["sentiment"].dropna().unique(),
            default=df["sentiment"].dropna().unique()
        )

        aspect_filter = st.multiselect(
            "Aspect",
            options=df["aspect"].dropna().unique(),
            default=df["aspect"].dropna().unique()
        )

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


# ANALYTICS & INSIGHTS DASHBOARD

elif menu == "📈 Analytics and Insights Dashboard":
    st.subheader("📈 Analytics and Insights Dashboard")

    st.markdown("## 📌 Overview of Customer Sentiment Trends")

    # Fetch all data from the database
    results = get_all_results()
    db_data = pd.DataFrame(results)

    # Fetch session data
    session_data = st.session_state.get("results_df", pd.DataFrame())

    # Ensure same structure
    required_cols = ["text", "sentiment", "aspect", "confidence", "timestamp"]
    for col in required_cols:
        if col not in session_data.columns:
            session_data[col] = None

    # Combine safely
    if not db_data.empty and not session_data.empty:
        data = pd.concat([db_data, session_data], ignore_index=True)
    elif not db_data.empty:
        data = db_data
    else:
        data = session_data

    if not data.empty:
        data["timestamp"] = pd.to_datetime(data["timestamp"])

        total_feedback = len(data)
        negative_rate = data["sentiment"].value_counts().get("negative", 0) / total_feedback * 100
        top_aspect = data["aspect"].value_counts().idxmax()
        avg_conf = data["confidence"].mean()

        # KPI metrics
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Feedback", total_feedback)
        kpi2.metric("Negative Rate", f"{negative_rate:.1f}%")
        kpi3.metric("Top Topic", top_aspect)
        kpi4.metric("Avg Confidence", f"{avg_conf * 100:.1f}%")


        # Sentiment Pie Chart
        st.subheader("Sentiment Distribution")

        # Clean data (VERY IMPORTANT)
        clean_data = data.copy()
        clean_data["sentiment"] = clean_data["sentiment"].astype(str).str.lower().str.strip()

        # Remove nulls
        clean_data = clean_data[clean_data["sentiment"].notna()]

        # Check if we have data to display
        if not clean_data.empty:
            pie_df = clean_data["sentiment"].value_counts().reset_index()
            pie_df.columns = ["sentiment", "count"]

            fig1 = px.pie(
                pie_df,
                names="sentiment",
                values="count",
                color="sentiment",
                color_discrete_map={
                    "negative": "#EF4444",   # red
                    "neutral": "#F59E0B",    # amber
                    "positive": "#10B981"    # green
                }
            )

            # Improve appearance
            fig1.update_traces(textposition='inside', textinfo='percent+label')
            fig1.update_layout(showlegend=True)

            st.plotly_chart(fig1, use_container_width=True)

        else:
            st.warning("No sentiment data available to display chart.")


        # Aspect Bar Chart
        st.subheader("Aspect Distribution")
        aspect_df = data["aspect"].value_counts().reset_index()
        aspect_df.columns = ["aspect", "count"]
        fig2 = px.bar(aspect_df, x="aspect", y="count")
        st.plotly_chart(fig2, use_container_width=True)

        # Recent Negative Feedback
        st.subheader("Recent Negative Feedback")
        negative = data[data["sentiment"] == "negative"].tail(5)
        for _, row in negative.iterrows():
            st.error(
                f"⚠ Message: {row['text']} | Aspect: {row['aspect']} | Confidence: {row['confidence']:.2f}"
            )

        # Automated Insights per Aspect
        st.subheader("Automated Insights")
        insights = []
        for aspect in data["aspect"].dropna().unique():
            aspect_df = data[data["aspect"] == aspect]
            neg_pct = aspect_df["sentiment"].value_counts().get("negative", 0) / len(aspect_df) * 100
            insights.append(f"- {aspect}: {neg_pct:.1f}% negative feedback")
        st.markdown("\n".join(insights))

        # Full CSV Download
        st.subheader("Download Full Analytics Data")
        full_csv = data.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Full Analytics CSV",
            full_csv,
            "full_analytics.csv",
            "text/csv"
        )

    else:
        st.info("No data available yet.")