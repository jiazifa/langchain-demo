from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain, SequentialChain
from langchain.memory import SimpleMemory

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0.2)    # type: ignore


def simple_demo():
    # 剧作家，先根据名字构造一段简介
    synopsis_template = """you are a playwright. Give the title of play, it is your job to write a synopsis for that title.
    Title: {title}
    Playwright: This is a synopsis for the above play:"""
    synopsis_prompt_template = PromptTemplate.from_template(synopsis_template)
    synopsis_chain = LLMChain(llm=llm, prompt=synopsis_prompt_template)

    # 评论家，根据简介，输出一段评价
    review_template = """you are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.
    Play Synopsis:
    {synopsis}
    Review from a New York Times play critic of the above play:"""
    review_prompt = PromptTemplate.from_template(review_template)
    review_chain = LLMChain(llm=llm, prompt=review_prompt)

    overall_chain = SimpleSequentialChain(
        chains=[synopsis_chain, review_chain], verbose=True
    )
    review = overall_chain.run("海滩上日落时的悲剧")
    print(review)


def demo2():
    # 剧作家，先根据名字构造一段简介, 包含了多个字段
    synopsis_template = """You are a playwright. Given the title of play and the era it is set in, it is your job to write a synopsis for that title.

    Title: {title}
    Era: {era}
    Actors: {actors}
    Playwright: This is a synopsis for the above play:"""
    synopsis_prompt_template = PromptTemplate(
        input_variables=["title", 'era', 'actors'], template=synopsis_template
    )
    synopsis_chain = LLMChain(
        llm=llm, prompt=synopsis_prompt_template, output_key="synopsis"
    )

    # 评论家，根据简介，输出一段评价
    review_template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

    Play Synopsis:
    {synopsis}
    Review from a New York Times play critic of the above play:"""
    review_prompt = PromptTemplate(
        input_variables=["synopsis"], template=review_template
    )
    review_chain = LLMChain(llm=llm, prompt=review_prompt, output_key="review")

    # 社会评论家
    social_template = """You are a social media manager for a theater company.  Given the title of play, the era it is set in, the date,time and location, the synopsis of the play, and the review of the play, it is your job to write a social media post for that play.

    Here is some context about the time and location of the play:
    Date and Time: {time}
    Location: {location}

    Play Synopsis:
    {synopsis}
    Review from a New York Times play critic of the above play:
    {review}

    Social Media Post:
    """
    social_prompt_template = PromptTemplate(
        input_variables=["synopsis", "review", "time", "location"],
        template=social_template
    )
    social_chain = LLMChain(
        llm=llm, prompt=social_prompt_template, output_key="social_post_text"
    )

    memory = SimpleMemory(
        memories={
            "time": "December 25th, 8pm PST",
            "location": "Theater in the Park"
        }
    )
    overall_chain = SequentialChain(
        memory=memory,
        chains=[synopsis_chain, review_chain, social_chain],
        input_variables=["era", "title", "actors"],
    # Here we return multiple variables
    # output_variables=["social_post_text"],
        verbose=True
    )
    review = overall_chain(
        {
            "title": "Tragedy at sunset on the beach",
            "era": "1840s",
            "actors": "Tim Robbins, Morgan Freeman, Bob Gunton, William Sadler",
        }
    )
    print(review)
    # {
    #     'title': 'Tragedy at sunset on the beach',
    #     'era': '1840s',
    #     'actors': 'Tim Robbins, Morgan Freeman, Bob Gunton, William Sadler',
    #     'time': 'December 25th, 8pm PST',
    #     'location': 'Theater in the Park',
    #     'social_post_text':
    #         "Don't miss out on the gripping drama, Tragedy at Sunset on the Beach, featuring an all-star cast including Tim Robbins, Morgan Freeman, Bob Gunton, and William Sadler. Set in the 1840s on a deserted beach, this play will leave you on the edge of your seat as tensions rise and secrets are revealed. The haunting score adds to the sense of foreboding, creating a palpable sense of unease that lingers long after the play has ended. The tragic and unexpected conclusion is both heartbreaking and thought-provoking, leaving audiences with much to ponder. Join us on December 25th at 8pm PST at the Theater in the Park for a must-see performance. #TragedyatSunsetontheBeach #ClassicDrama #AllStarCast #TheaterinthePark #MustSeePerformance"
    # }


if __name__ == "__main__":
    demo2()
