import gradio as gr
from transformers import pipeline

# Loading model from hugging face
sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")


def analyze_sentiment(text):
    
    result = sentiment_model(text)
    label = result[0]['label']
    score = result[0]['score']
    
    # Labelling the cx feedback
    label_mapping = {
        '1 star': 'Very Negative',
        '2 stars': 'Negative',
        '3 stars': 'Neutral',
        '4 stars': 'Positive',
        '5 stars': 'Very Positive'
    }
    sentiment_result = f"**Sentiment**: {label_mapping[label]}\n**Confidence**: {score:.2f}"
    return sentiment_result

# Custom CSS for examples
css = """
body {background-color: #f4f4f4; font-family: 'Arial'; color: #333;}
input, textarea {border-radius: 10px; border: 2px solid #999; padding: 10px; background-color: #fff; color: #333;}
button {background-color: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 10px; cursor: pointer;}
button:hover {background-color: #2980b9;}
.output-text {font-size: 18px; color: #333;}
footer {display: none !important;}
/* Dark theme */
@media (prefers-color-scheme: dark) {
    body {background-color: #2c3e50; color: #ecf0f1;}
    input, textarea {background-color: #34495e; color: #ecf0f1; border: 2px solid #999;}
    button {background-color: #2980b9;}
    button:hover {background-color: #1abc9c;}
    .output-text {color: #ecf0f1;}
}
"""

interface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(
        lines=5, 
        placeholder="Enter a marketplace review or sentence here...",
        label="Input Review",
    ),
    outputs=gr.Markdown(),
    title="Sentiment Reveal",
    description=(
        "Analyze the sentiment of product reviews in English, Dutch, German, French, Italian, and Spanish. Focused Sentiment Analysis for eCommerce."
    ),
    examples=[["This product is amazing! I highly recommend it."],
              ["I'm very disappointed with this purchase."],
              ["The product was okay, not great but not terrible."]],
    allow_flagging="never",
    css=css,
)


interface.launch(share=True)
