import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import re
import string
import nltk
from sklearn.model_selection import train_test_split
import io
import plotly.express as px
from streamlit_extras.let_it_rain import rain
from wordcloud import WordCloud

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# Function to clean and preprocess text
def clean_text(text):
    if isinstance(text, str):
        return re.sub('[^a-zA-Z]', ' ', text).lower()
    elif text is not None:
        return str(text)
    else:
        return ""

# Function to count punctuations
def count_punct(review):
    review = review.strip() # Menghapus spasi ekstra di awal dan akhir
    if not review: # Jika review menjadi kosong
        return 0.0

    total_characters = len(review) - review.count(" ")
    if total_characters <= 1:
        return 0.0

    count = sum([1 for char in review if char in string.punctuation])
    return round(count / total_characters, 3) * 100

# Function to tokenize text
def tokenize_text(text):
    tokenized_text = text.split() # Mecahin jadi token-token
    return tokenized_text 

# Load stopwords from a text file
with open('data/stopwordsID.txt', 'r', encoding='utf-8') as stopwords_file:
    stopwords_set = set(stopwords_file.read().splitlines())
    
# Function to lemmatize text and remove stopwords
# mengubah kata menjadi bentuk dasarnya
def lemmatize_text(token_list, stopwords_set):
    return " ".join([lemmatizer.lemmatize(token) for token in token_list if not token in stopwords_set])

# Function to analyze sentiment for a given text
def analyze_sentiment(text):
    cleaned_input = clean_text(text)
    lemmatized_input = lemmatize_text(tokenize_text(cleaned_input), stopwords_set)

    input_vector = tfidf.transform([lemmatized_input]) # mengubah teks menjadi bentuk numerik
    prediction = classifier.predict(input_vector)[0]  #  melakukan prediksi sentimen berdasarkan vektor fitur

    return cleaned_input, prediction

# Load and preprocess the dataset
df = pd.read_csv("data/dataset_sentiment.csv", sep=';', encoding="ISO-8859-1")

# START Preprocessing
df['cleaned_text'] = df['Text Tweet'].apply(lambda x: clean_text(x))
df['label'] = df['Sentiment'].map({'negative': 0, 'positive': 1})
df['review_len'] = df['cleaned_text'].apply(lambda x: len(str(x).split()))
df['punct'] = df['cleaned_text'].apply(lambda x: count_punct(x))

#  database leksikal untuk lemmatisasi
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = nltk.stem.WordNetLemmatizer()
df['tokens'] = df['cleaned_text'].apply(lambda x: tokenize_text(x)) # menambahkan kolom baru dgn apply
df['lemmatized_review'] = df['tokens'].apply(lambda x: lemmatize_text(x, stopwords_set)) # menambahkan kolom baru dgn apply

# menghapus nilai yang hilang (NaN) di kolom label secara langsung
df.dropna(subset=['label'], inplace=True) 

# END Preprocessing

# Feature Extraction (TF-IDF)
X = df['lemmatized_review']
y = df['label']
# pemisahan data menjadi data pelatihan (train) dan pengujian (test)
# 30% dari data akan digunakan sebagai data pengujian, sementara 70% akan digunakan sebagai data pelatihan
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

tfidf = TfidfVectorizer(max_df=0.5, min_df=2) # untuk mengonversi teks menjadi vektor fitur TF-IDF.
tfidf_train = tfidf.fit_transform(X_train) # mengonversi data pelatihan menjadi vektor fitur TF-IDF
tfidf_test = tfidf.transform(X_test) # mengonversi data pengujian menjadi vektor fitur TF-IDF

# PREDICTION
# Using SVM Classifier
classifier = SVC(kernel='linear', random_state=10)
# Model ini akan mempelajari pola dalam data pelatihan 
classifier.fit(tfidf_train, y_train) # melatih model SVM dengan menggunakan data pelatihan.

# Streamlit app
def main():
    st.markdown("<h1 style='text-align: center;'>Aplikasi Sentiment Analysis</h3>", unsafe_allow_html=True)
    # Menu Utama - Analisis Teks
    menu = st.sidebar.selectbox(f"**MENU**", ["Home", "Analisis Teks", "Analisis File Excel"])
    
    # Menu Landing Page - Home
    if menu == "Home":
        st.image('img/sentimentPict.jpeg', use_column_width=True)

        st.markdown("### Selamat datang di Aplikasi Analisis Sentimen! 🥳")
        st.markdown("Aplikasi ini memungkinkan Anda untuk menganalisis sentimen dari teks yang Anda masukkan.Sebelumnya, apakah Anda sudah mengerti arti dari Sentiment analysis?")
        st.markdown("Jika belum, berikut penjelasan singkat mengenai Sentiment analysis: ")                                                                                                                                                                                  
        st.markdown(" **Warta Ekonomi, Jakarta** - Sentiment analysis (atau penggalian opini) adalah teknik pemrosesan bahasa secara alami yang digunakan untuk menentukan apakah data tersebut positif, negatif, atau netral. Sentiment analysis sering dilakukan pada data tekstual untuk membantu bisnis dalam memantau sentimen pelanggan terhadap brand dan produk dalam bentuk feedback, dan memahami segala kebutuhan pelanggan.")
        st.markdown("Penggunaan Sentiment Analysis:")
        st.markdown("**1. Analisis Sentimen Sosial Media:** Monitoring sentimen publik terhadap merek, produk, atau layanan melalui platform media sosial seperti Twitter, Facebook, dan Instagram. Selain itu juga dapat memahami reaksi pengguna terhadap peristiwa penting atau berita terkini.")
        st.markdown("**2. Analisis Pasar Keuangan:** Menganalisis sentimen pasar saham berdasarkan berita keuangan dan laporan perusahaan sehingga dapat membantu investor dalam mengambil keputusan berdasarkan analisis sentimen pasar.")
        st.markdown("**3. Filtering dan Klasifikasi Teks:** Mengidentifikasi konten berbahaya atau tidak diinginkan dalam platform online.")
    
    elif menu == "Analisis Teks":
        st.sidebar.header('Masukan Pengguna')

        # Text input from the user
        user_input = st.sidebar.text_area('Masukkan ulasan Anda:', '')

        if st.sidebar.button('Menganalisa'):
            cleaned_input, prediction = analyze_sentiment(user_input)

            # Count additional features: review length and percentage of punctuation
            review_length = len(cleaned_input.split())
            punctuation_percentage = count_punct(user_input)


            st.markdown("")
            st.markdown("")
            st.markdown("")
            
            st.info(f'**HASIL SENTIMEN:**')
            st.write(f"**Teks Awal**: {user_input}")
            # Determine overall sentiment based on sentiment prediction
            overall_sentiment = "Positif 😃" if prediction == 1 else "Negatif 😠"
            st.write(f'**Sentiment:**', overall_sentiment)
            # Display the additional information
            st.write('Panjang Ulasan:', review_length)
            st.write('Persentase Tanda Baca:', punctuation_percentage)
            
            st.markdown("")
            st.markdown("")
            st.markdown("")
            
            # Display sentiment-specific messages and animations
            if prediction == 1:  # Positive sentiment
                st.success(f'**Selamat, ulasan kamu memiliki Sentimen Positif**')
                rain(emoji="🎈", font_size=54, falling_speed=2, animation_length=1)
            else:  # Negative sentiment
                st.warning(f'**Yah.., ulasan kamu memiliki **Sentimen Negatif**, coba tulis ulasan lainnya lagi yaa 😥**')
                rain(emoji="😥 ", font_size=54, falling_speed=2, animation_length=1)
                
    elif menu == "Analisis File Excel":
        with st.sidebar:
            upl = st.file_uploader('Upload file (excel)', type=["xlsx"])
        
        # Function to create and display a word cloud
        def create_word_cloud(text, stopwords):
            # Generate the word cloud with stopwords removed
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                stopwords=stopwords  # Use the stopwords loaded from the text file
            ).generate(text)

            st.image(wordcloud.to_image(), use_column_width=True)


        if upl:
            df = pd.read_excel(upl)

            def analyze_and_label_sentiment(text):
                cleaned_input, prediction = analyze_sentiment(text)

                if prediction == 1:
                    return "Positif 😃"
                elif prediction == 0:
                    return "Negatif 😠"

            df['analysis'] = df['Comment'].apply(analyze_and_label_sentiment)
            
            st.markdown("")
            st.markdown("")
            st.markdown("")
            st.write(df.head(10))
        
            def convert_df_excel(df):
                # Create a file-like object in memory
                excel_buffer = io.BytesIO()
                # Save the DataFrame to the file-like object
                df.to_excel(excel_buffer, index=False)
                # Get the bytes from the file-like object
                excel_data = excel_buffer.getvalue()
                return excel_data

            excel_data = convert_df_excel(df)

            def convert_df_csv(df):
                # Create a file-like object in memory
                csv_buffer = io.BytesIO()
                # Save the DataFrame to the file-like object
                df.to_csv(csv_buffer, index=False)
                # Get the bytes from the file-like object
                csv_data = csv_buffer.getvalue()
                return csv_data  # Define and return the csv_data variable here

            csv_data = convert_df_csv(df)  # Assign the csv data to the csv_data variable

            col1_width = 200
            col2_width = 800

            col1, col2 = st.columns([col1_width, col2_width])
            with col1:
                st.download_button(
                    label="Download data as Excel",
                    data=excel_data,
                    file_name='sentiment.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                )
            with col2:
                st.download_button(
                    label="Download data as CSV",
                    data=csv_data,
                    file_name='sentiment.csv',
                    mime='text/csv',  # Correct MIME type for CSV
                )
            
            st.markdown("")
            st.markdown("")
            st.markdown("")
            
            st.info("### Word Cloud Visualization")
            # Load stopwords from a text file
            with open('data/stopwordsID.txt', 'r', encoding='utf-8') as stopwords_file:
                stopwords_set = set(stopwords_file.read().splitlines())

            # Combine all text from the 'Comment' column for word cloud
            text = ' '.join(df['Comment'].astype(str))
            create_word_cloud(text, stopwords_set)
            
            st.markdown("")
            st.markdown("")
            st.markdown("")
    
            col1_width = 300
            col2_width = 700

            col1, col2 = st.columns([col1_width, col2_width])
            with col1:
                st.info("### Proses Analisis Sentimen")
                # Hitung jumlah masing-masing sentimen
                sentiment_counts = df['analysis'].value_counts()
                total_sentiments = len(df)
    
                st.write("Hasil Sentiment:")
                for sentiment, count in sentiment_counts.items():
                    st.write(f"{sentiment}: {count}")
                    st.write(f"Total: {total_sentiments}")
            with col2:
                st.info("### Visualisasi Sentimen")
                st.write("Distribusi Sentimen:")
                
                # Create a pie chart using plotly
                fig = px.pie(sentiment_counts, names=sentiment_counts.index, values=sentiment_counts.values)
        
                # Customize the pie chart
                fig.update_traces(textposition='inside', textinfo='percent+label')

                # Display the pie chart
                st.plotly_chart(fig)

if __name__ == '__main__':
    main()
