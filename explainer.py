from groq import Groq
import os

# groq for ai explanations of results
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def explain_anomalies(summary, columns):
    methods = summary["methods"]

    prompt = f"""Here are anomaly detection results from a dataset scan:

Analyzed {summary['total_rows']} rows across columns: {', '.join(columns)}
Found {summary['total_anomalies']} anomalies ({summary['anomaly_percentage']}% of data)

Breakdown by method:
- Isolation Forest flagged {methods['isolation_forest']['anomalies']}
- Autoencoder flagged {methods['autoencoder']['anomalies']}
- Z-Score flagged {methods['zscore']['anomalies']}

Avg anomaly score: {summary['avg_anomaly_score']}

Explain what these results mean, which method found most and why, how reliable the combined results are, and what to investigate next. Keep it under 8 sentences."""

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "you are a data analyst. explain anomaly detection results clearly and practically."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=500
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"could not generate explanation: {str(e)}"
