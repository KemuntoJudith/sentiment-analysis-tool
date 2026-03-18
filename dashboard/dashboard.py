# Sentiment Monitoring Dashboard - Streamlit Frontend
# Supports Local Mode (models & DB locally) and Remote API Mode


# ENVIRONMENT SETUP
from dotenv import load_dotenv
load_dotenv()

import sys
import os

# Add project root to sys.path so 'app' can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Verify that Python can see 'app'
print("Project root added to sys.path:", PROJECT_ROOT)
print("sys.path[0]:", sys.path[0])


# IMPORTS
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import plotly.io as pio
import time
import io
import requests
from app.utils.db import save_result
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


# Local models & DB
from app.models.finbert_model import predict_sentiment as predict_sentiment_local
from app.models.absa_model import analyze_aspects as analyze_aspects_local
from app.utils.db import call_register, call_login, save_result

# Password hashing
from passlib.hash import bcrypt

# CONFIG
API_BASE = os.getenv("API_BASE")
if API_BASE == "":
    API_BASE = None


# WRAPPER FUNCTIONS
def call_predict_sentiment(text, user_id):
    """Handles both API and local inference safely"""
    if API_BASE:
        response = requests.post(
            f"{API_BASE}/predict-sentiment",
            json={"text": text, "user_id": user_id},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    else:
        sentiment_result = predict_sentiment_local(text)
        aspect_result = analyze_aspects_local(text)

        # aspect handling
        aspects = aspect_result.get("aspects", [])
        aspect = aspects[0].get("aspect", "general") if aspects else "general"

        return {
            "text": text,
            "sentiment": sentiment_result.get("sentiment", "neutral"),
            "confidence": sentiment_result.get("confidence", 0.0),
            "aspect": aspect
        }


# PDF GENERATOR
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

    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Aspect Insights:", styles["Heading2"]))

    for aspect in data["aspect"].dropna().unique():
        subset = data[data["aspect"] == aspect]
        neg_pct = subset["sentiment"].value_counts().get("negative", 0) / len(subset) * 100
        elements.append(Paragraph(f"{aspect}: {neg_pct:.1f}% negative", styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer


# SESSION STATE INIT
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


# AUTH UI (LOGIN + REGISTER)
if not st.session_state["logged_in"]:
    with st.sidebar:
        tab1, tab2 = st.tabs(["Login", "Register"])

        # LOGIN
        with tab1:
            st.subheader("Login")
            username_input = st.text_input("Username", key="login_user")
            password_input = st.text_input("Password", type="password", key="login_pass")

            if st.button("Login", key="btn_login"):
                try:
                    user_data = call_login(username_input, password_input)
                    st.session_state["logged_in"] = True
                    st.session_state["user_id"] = user_data["user_id"]
                    st.session_state["username"] = user_data["username"]
                    st.success(f"Logged in as {user_data['username']}")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

                # REGISTER
            with tab2:
                st.subheader("Register")
                new_username = st.text_input("New Username", key="reg_user")
                new_password = st.text_input("New Password", type="password", key="reg_pass")

                if st.button("Register", key="btn_register"):
                    try:
                        # Truncate password to 72 characters before sending
                        safe_password = new_password[:72]
                        call_register(new_username, safe_password)
                        st.success("Registration successful! You can now log in.")

                        # Clear inputs
                        st.session_state["reg_user"] = ""
                        st.session_state["reg_pass"] = ""
                        st.experimental_rerun()

                    except Exception as e:
                        st.error(str(e))


# DASHBOARD
if st.session_state["logged_in"]:
    st.title(f"Sentiment Monitoring Dashboard - {st.session_state['username']}")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Single Message", "Batch Upload", "Reports & Analytics", "Live Feedback"]
    )


    # SINGLE MESSAGE
    with tab1:
        st.subheader("Analyze a Single Message")

        with st.form("single_message_form"):
            user_input = st.text_area("Enter customer feedback")
            submitted = st.form_submit_button("Analyze Sentiment")

            if submitted and user_input.strip():
                try:
                    result = call_predict_sentiment(
                        user_input, st.session_state["user_id"]
                    )
                    result["timestamp"] = datetime.now()


                    st.session_state["results_df"] = pd.concat(
                        [st.session_state["results_df"], pd.DataFrame([result])],
                        ignore_index=True
                    )

                    # --- SAVE TO DATABASE ---
                    try:
                        save_result(
                            text=result["text"],
                            aspect=result["aspect"],
                            sentiment=result["sentiment"],
                            confidence=result["confidence"],
                            user_id=st.session_state["user_id"]
                        )
                    except Exception as e:
                        st.warning(f"Could not save result to DB: {e}")

                    st.success("Analysis Complete")
                    st.write("**Sentiment:**", result["sentiment"])
                    st.write("**Confidence:**", f"{result['confidence']:.2f}")
                    st.write("**Aspect:**", result["aspect"])

                except Exception as e:
                    st.error(f"Failed to analyze: {e}")


    # BATCH UPLOAD
    def analyze_batch(df_upload):
        results = []
        progress_bar = st.progress(0)

        for i, text in enumerate(df_upload["text"]):
            try:
                res = call_predict_sentiment(
                    str(text), st.session_state["user_id"]
                )
                res["timestamp"] = datetime.now()
                results.append(res)

                # --- SAVE TO DATABASE ---
                try:
                    save_result(
                        text=res["text"],
                        aspect=res["aspect"],
                        sentiment=res["sentiment"],
                        confidence=res["confidence"],
                        user_id=st.session_state["user_id"]
                    )
                except Exception as e:
                    st.warning(f"Could not save row {i} to DB: {e}")

            except Exception as e:
                st.warning(f"Skipping row {i} due to error: {e}")
                results.append({
                    "text": text,
                    "sentiment": None,
                    "aspect": None,
                    "confidence": None,
                    "timestamp": datetime.now()
                })

            progress_bar.progress((i + 1) / len(df_upload))
            time.sleep(0.1)

        return pd.DataFrame(results)

    with tab2:
        st.subheader("Batch Upload Feedback")

        uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

        if uploaded_file:
            df_upload = (
                pd.read_csv(uploaded_file)
                if uploaded_file.name.endswith(".csv")
                else pd.read_excel(uploaded_file)
            )

            if "text" not in df_upload.columns:
                st.error("File must contain 'text' column.")
            else:
                if st.button("Analyze Uploaded File"):
                    batch_results = analyze_batch(df_upload)

                    st.session_state["results_df"] = pd.concat(
                        [st.session_state["results_df"], batch_results],
                        ignore_index=True
                    )

                    st.dataframe(batch_results)

                    csv = batch_results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Batch Results",
                        csv,
                        "batch_results.csv",
                        "text/csv"
                    )
    

    # REPORTS & ANALYTICS
    with tab3:
        st.subheader("Reports & Analytics")

        df = st.session_state["results_df"]

        if not df.empty:
            col1, col2, col3 = st.columns(3)

            start_date = col1.date_input(
                "Start Date", df["timestamp"].min().date()
            )
            end_date = col2.date_input(
                "End Date", df["timestamp"].max().date()
            )

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


    # LIVE FEEDBACK
    with tab4:
        st.subheader("Live Feedback Simulation (CSV Input)")

        uploaded_live_file = st.file_uploader(
            "Upload CSV for live feed", type=["csv"], key="live_csv"
        )

        if uploaded_live_file:
            live_df = pd.read_csv(uploaded_live_file)

            if "text" not in live_df.columns:
                st.error("CSV must contain 'text' column.")
            else:
                if st.button("Start Live Stream"):
                    st.session_state["live_running"] = True

                if st.button("Stop Live Stream"):
                    st.session_state["live_running"] = False

                if st.session_state["live_running"]:
                    st.info("Live streaming active (every 2 seconds)")

                    for _, row in live_df.iterrows():
                        if not st.session_state["live_running"]:
                            break

                        text = str(row["text"])

                        try:
                            result = call_predict_sentiment(
                                text, st.session_state["user_id"]
                            )
                            result["timestamp"] = datetime.now()

                            st.session_state["results_df"] = pd.concat(
                                [st.session_state["results_df"], pd.DataFrame([result])],
                                ignore_index=True
                            )

                            try:
                                save_result(
                                    text=result["text"],
                                    aspect=result["aspect"],
                                    sentiment=result["sentiment"],
                                    confidence=result["confidence"],
                                    user_id=st.session_state["user_id"]
                                )
                            except Exception as e:
                                st.warning(f"Could not save live feedback to DB: {e}")

                            st.success(f"New feedback analyzed: {text}")

                        except Exception as e:
                            st.error(f"Failed: {text} | {e}")

                        time.sleep(2)



# ANALYTICS
st.subheader("Analytics & Insights")

data = st.session_state["results_df"]

if not data.empty:
    total_feedback = len(data)
    negative_rate = (
        data["sentiment"].value_counts().get("negative", 0)
        / total_feedback * 100
    )
    top_aspect = data["aspect"].value_counts().idxmax()
    avg_conf = data["confidence"].mean()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Feedback", total_feedback)
    kpi2.metric("Negative Rate", f"{negative_rate:.1f}%")
    kpi3.metric("Top Topic", top_aspect)
    kpi4.metric("Avg Confidence", f"{avg_conf:.2f}")


    # SENTIMENT PIE
    st.subheader("Sentiment Distribution")
    pie_df = data["sentiment"].value_counts().reset_index()
    pie_df.columns = ["sentiment", "count"]

    fig1 = px.pie(pie_df, names="sentiment", values="count")
    st.plotly_chart(fig1, use_container_width=True)

    # SAFE DOWNLOAD
    try:
        buf1 = io.BytesIO()
        fig1.write_image(buf1, format="png")
        st.download_button(
            "Download Sentiment Pie",
            buf1.getvalue(),
            "sentiment_pie.png",
            "image/png"
        )
    except Exception:
        st.info("Install 'kaleido' to enable image downloads.")


    # ASPECT BAR
    st.subheader("Aspect Distribution")
    aspect_df = data["aspect"].value_counts().reset_index()
    aspect_df.columns = ["aspect", "count"]

    fig2 = px.bar(aspect_df, x="aspect", y="count")
    st.plotly_chart(fig2, use_container_width=True)

    # SAFE DOWNLOAD
    try:
        buf2 = io.BytesIO()
        fig2.write_image(buf2, format="png")
        st.download_button(
            "Download Aspect Bar",
            buf2.getvalue(),
            "aspect_bar.png",
            "image/png"
        )
    except Exception:
        st.info("Install 'kaleido' to enable image downloads.")


    # RECENT NEGATIVE
    st.subheader("Recent Negative Feedback")
    negative = data[data["sentiment"] == "negative"].tail(5)
    for _, row in negative.iterrows():
        st.error(
            f"⚠ Message: {row['text']} | Aspect: {row['aspect']} | Confidence: {row['confidence']:.2f}"
        )


    # AUTOMATED INSIGHTS
    st.subheader("Automated Insights")
    insights = []

    for aspect in data["aspect"].dropna().unique():
        aspect_df = data[data["aspect"] == aspect]
        neg_pct = (
            aspect_df["sentiment"].value_counts().get("negative", 0)
            / len(aspect_df) * 100
        )
        insights.append(f"- {aspect}: {neg_pct:.1f}% negative feedback")

    st.markdown("\n".join(insights))


    # FULL CSV DOWNLOAD
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