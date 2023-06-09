from langchain.vectorstores.base import Document


def test_v_database():
    from app.store import VDatabase

    documents = [
        Document(
            page_content=
            """I am Annie, I am 1.65 meters tall and weigh 50 kilograms. I am a person full of energy and curiosity, who likes to explore new things and pursue different lifestyles.

I was born into an ordinary family, and both of my parents are normal office workers. From a young age, I had a strong interest in art and music, and often went to various exhibitions and concerts with my parents. This also made me a person with a vivid imagination and creativity.

In school, I was very active and often participated in various activities and competitions. I won many awards in music competitions and also excelled in art and writing. These experiences made me more confident and determined to pursue my dreams.

During my university years, I chose to study design. Here, I learned a lot of knowledge and skills about design, and also made many like-minded friends. During my time in school, I also participated in many design competitions and activities, constantly improving my abilities and level.

After graduation, I started my career as a designer. I worked for a well-known design company, responsible for designing various products and brand images. My design style is unique and loved by customers and users. I also started my own personal brand, launching a series of design works that have been recognized by the market.

In addition to work, I also like to travel and take photos. I have been to many countries and regions, taking many beautiful landscape and cultural photos. These experiences have broadened my horizons and made me more passionate about life.

Overall, I am a person full of passion and creativity, who likes to challenge myself and constantly explore new things. I believe that as long as I insist on my dreams, keep working hard and learning, I will definitely achieve my goals.
            """,
            metadata={"description": "Basic personal information"}
        )
    ]
    db = VDatabase(persist_directory="tmp", collection_name="test", embedding=None)
    docs = db.query(
        query=
        "I am Annie, I am 1.65 meters tall and weigh 50 kilograms. I am a person full of energy and curiosity, who likes to explore new things and pursue different lifestyles.",
        search_type="similarity"
    )
    if len(docs) == 0:
        _ = db.insert(documents)
        docs = db.query(query="test", search_type="similarity")
    assert len(docs) == 1
    doc = docs[0]
    assert doc.metadata["description"] == "Basic personal information"
