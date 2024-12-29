#import nltk
#nltk.download('brown')
#import nltk
#nltk.download('wordnet')
#!unzip /usr/share/nltk_data/corpora/wordnet.zip -d /usr/share/nltk_data/corpora/
from nltk.stem import WordNetLemmatizer
import torch
import re
def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Remove HTML tags
    text = re.sub(r"<.*?>", '', text)
    # Remove special characters and numbers
    text = re.sub(r"[^A-Za-z\s]", '', text)
    # Remove extra whitespace
    text = re.sub(r"\s+", ' ', text).strip()
    return text

lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def preprocess_text(text):
    # Clean the text
    text = clean_text(text)
    # Lowercase
    text = text.lower()
    # Lemmatize text
    text = lemmatize_text(text)
    return text





def split_into_chunks(text, tokenizer, max_tokens=512):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))
    return chunks



def predict_article(article, model, tokenizer, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    article=preprocess_text(article)
    # Split article into chunks
    paragraphs = split_into_chunks(article, tokenizer)
    ai_count = 0
    human_count = 0

    for paragraph in paragraphs:
        inputs = tokenizer(paragraph, return_tensors="pt", truncation=True, max_length=512)

        # Move inputs to the same device as the model
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            if predicted_class == 1:  # AI-Generated
                ai_count += 1
            else:  # Human-Generated
                human_count += 1

    # Aggregate results
    '''
    if ai_count >= human_count:
        return "AI-Generated", ai_count, human_count
    else:
        return "Human-Generated", ai_count, human_count
    '''
    if human_count >= 1:
        return "Human-Generated", ai_count, human_count
    else:
        return "AI-Generated", ai_count, human_count
    