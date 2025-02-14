# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Tuple

from pikerag.prompts import BaseContentParser, CommunicationProtocol, MessageTemplate
from pikerag.utils.json_parser import parse_json


"""
Demonstration used here comes from IRCoT's preprocessed gold_with_3_distractors_context_cot_qa_codex.txt file.
Consisting of first two 2-hops, first one 3-hops, first one 4-hops examples.
"""
ircot_template = MessageTemplate(
    template=[
        ("system", "You are a helpful AI assistant good at question-answering."),
        ("user", """
# Task
Your task is to output either a continuous reasoning sentence or the final answer to the given question. Four demonstrations would be provided, followed by the question to be answered and the reference context you can refer to.

# Demonstration
Question: When was Neville A. Stanton's employer founded?
Reference context:
  1. Title: The Last Horse. The Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernán Gómez.
  2. Title: Southampton. The University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. The university considers itself one of the top 5 research universities in the UK. The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.
  3. Title: Stanton Township, Champaign County, Illinois. Stanton Township is a township in Champaign County, Illinois, USA. As of the 2010 census, its population was 505 and it contained 202 housing units.
  4. Title: Neville A. Stanton. Neville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. He has been published in academic journals including "Nature". He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.
  5. Title: Finding Nemo. Finding Nemo Theatrical release poster Directed by Andrew Stanton Produced by Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds Story by Andrew Stanton Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe Music by Thomas Newman Cinematography Sharon Calahan Jeremy Lasky Edited by David Ian Salter Production company Walt Disney Pictures Pixar Animation Studios Distributed by Buena Vista Pictures Distribution Release date May 30, 2003 (2003 - 05 - 30) Running time 100 minutes Country United States Language English Budget $94 million Box office $940.3 million
Rationale: The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862.
Answer: 1862

Question: What is the headquarters for the organization who sets the standards for ISO 21500?
Reference Context:
  1. Title: ISO 21500. ISO 21500:2012, Guidance on Project Management, is an international standard developed by the International Organization for Standardization, or ISO starting in 2007 and released in 2012. It was intended to provide generic guidance, explain core principles and what constitutes good practice in project management. The ISO technical committee dealing with project management, ISO/PC 236 was held by the American National Standards Institute (ANSI) which had approved four standards that used PMI materials. one of which was ANSI/PMI 99-001-2008, A Guide to the Project Management Body of Knowledge - 4th Edition (PMI BoK® Guide - 4th Edition) (revision and re-designation of ANSI/PMI 99-001-2004): 11/20/2008.
  2. Title: ISO 3166-2:GH. ISO 3166-2:GH is the entry for Ghana in ISO 3166-2, part of the ISO 3166 standard published by the International Organization for Standardization (ISO), which defines codes for the names of the principal subdivisions (e.g., provinces or states) of all countries coded in ISO 3166-1.
  3. Title: ISO 4031. ISO 4031 is an international standard first issued in 1978 by the International Organization for Standardization. It defined the representation of local time differentials, commonly referred to as time zones. It has since been superseded by a newer standard, ISO 8601. This newer standard sets out the current formats for local time differentials and so ISO 4031 is no longer in use.
  4. Title: ISO/TC 68. ISO/TC 68 is a technical committee formed within the International Organization for Standardization (ISO), of Geneva, Switzerland, tasked with developing and maintaining international standards covering the areas of banking, securities, and other financial services. As the standards organization under ISO responsible for the development of all international financial services standards, ISO/TC 68 plays a key role in the development and adoption of new technologies in the banking, brokerage and insurance industries. Many of its current work projects involve developing ecommerce standards such as better online security for financial transactions, XML standards for financial transactions and standards to reduce the cost and delays of international financial transactions. The membership of ISO/TC 68, consists of more than 30 organizations assigned by participating national standards bodies plus additional international standards development organizations that work collaboratively toward global financial services standards development.
  5. Title: ISO 3166-1. ISO 3166-1 is part of the ISO 3166 standard published by the International Organization for Standardization (ISO), and defines codes for the names of countries, dependent territories, and special areas of geographical interest. The official name of the standard is "Codes for the representation of names of countries and their subdivisions – Part 1: Country codes". It defines three sets of country codes:
Rationale: The standards for ISO 21500 were set by International Organization for Standardization. The International Organization for Standardization has headquarters in Geneva.
Answer: Geneva

Question: In which county was the birthplace of the Smoke in tha City performer?
Reference Context:
  1. Title: Cherokee City, Arkansas. Cherokee City is an unincorporated census-designated place in Benton County, Arkansas, United States. As of the 2010 census, its population is 72. It is the location of (or is the nearest community to) Coon Creek Bridge, which is located on Cty Rd. 24 and is listed on the National Register of Historic Places. The community was named for the Cherokee Indians, since the Trail of Tears crossed the landscape when the Cherokee migrated west to Indian territory, now Oklahoma in the late 1830s. The town is about 5 miles east of Oklahoma and 4 miles south of the Missouri state line.
  2. Title: Compton, California. Compton is a city in southern Los Angeles County, California, United States, situated south of downtown Los Angeles. Compton is one of the oldest cities in the county and on May 11, 1888, was the eighth city to incorporate. As of the 2010 United States Census, the city had a total population of 96,456. It is known as the "Hub City" due to its geographic centrality in Los Angeles County. Neighborhoods in Compton include Sunny Cove, Leland, Downtown Compton, and Richland Farms. The city is generally a working class city with some middle-class neighborhoods, and is home to a relatively young population, at an average 25 years of age, compared to the American median age of 38 (based on 2018 data).
  3. Title: MC Eiht. Aaron Tyler (born May 22, 1971), better known by his stage name MC Eiht, is an American rapper and actor. Many of his songs are based on his life in Compton. His stage name was partly inspired by the numeral in KRS-One's name. He chose Eiht for its links to "hood culture", including Olde English 800 (8 Ball) and .38 caliber firearms. He is the "de facto" leader of West Coast hip hop group Compton's Most Wanted, which also included fellow Compton-based rappers Boom Bam, Tha Chill, DJ Mike T, DJ Slip and Ant Capone. He is also known for his role as A-Wax in the 1993 film "Menace II Society".
  4. Title: Smoke in tha City. Smoke in tha City is the ninth studio album by American rapper MC Eiht, released September 14, 2004 on Factor House Records. It was produced by Black C-Zer and Quincy Miller. The album featuring guest performances by West Coast heavy-weights: RBX, Spice 1, Kokane, Jayo Felony and Daz Dillinger.
  5. Title: Beyoncé. On January 7, 2012, Beyoncé gave birth to her first child, a daughter, Blue Ivy Carter, at Lenox Hill Hospital in New York. Five months later, she performed for four nights at Revel Atlantic City's Ovation Hall to celebrate the resort's opening, her first performances since giving birth to Blue Ivy.
  6. Title: Olsztyn Voivodeship. Olsztyn Voivodeship () was an administrative division and unit of local government in Poland in the years 1945-75, and a new territorial division between 1975–1998, superseded by Warmian-Masurian Voivodeship. Its capital city was Olsztyn.
Rationale: The performer of Smoke in tha City is MC Eiht. MC Eiht's birthplace is Compton. Compton is located in the county of Los Angeles County.
Answer: Los Angeles County

Question: What weekly publication in the Connecticut city with the most Zagat rated restaurants is issued by university of America-Lite: How Imperial Academia Dismantled Our Culture's author?
Reference Context:
  1. Title: New Haven, Connecticut. New Haven is served by the daily New Haven Register, the weekly "alternative" New Haven Advocate (which is run by Tribune, the corporation owning the Hartford Courant), the online daily New Haven Independent, and the monthly Grand News Community Newspaper. Downtown New Haven is covered by an in-depth civic news forum, Design New Haven. The Register also backs PLAY magazine, a weekly entertainment publication. The city is also served by several student-run papers, including the Yale Daily News, the weekly Yale Herald and a humor tabloid, Rumpus Magazine. WTNH Channel 8, the ABC affiliate for Connecticut, WCTX Channel 59, the MyNetworkTV affiliate for the state, and Connecticut Public Television station WEDY channel 65, a PBS affiliate, broadcast from New Haven. All New York City news and sports team stations broadcast to New Haven County.
  2. Title: Imperial College London. Imperial College Union, the students' union at Imperial College, is run by five full-time sabbatical officers elected from the student body for a tenure of one year, and a number of permanent members of staff. The Union is given a large subvention by the university, much of which is spent on maintaining around 300 clubs, projects and societies. Examples of notable student groups and projects are Project Nepal which sends Imperial College students to work on educational development programmes in rural Nepal and the El Salvador Project, a construction based project in Central America. The Union also hosts sports-related clubs such as Imperial College Boat Club and Imperial College Gliding Club.
  3. Title: The End of Education. The End of Education is a book by Neil Postman about public education in America. The use of the word "end" in the title has two meanings: primarily, as a synonym for "purpose", but also as a prediction about the future of public schools if they do not successfully identify and communicate a convincing purpose for their existence within our culture.
  4. Title: America-Lite. America-Lite: How Imperial Academia Dismantled Our Culture (and Ushered in the Obamacrats) is a 2012 book by David Gelernter, published by Encounter Books.
  5. Title: David Gelernter. David Hillel Gelernter (born March 5, 1955) is an American artist, writer, and professor of computer science at Yale University. He is a former national fellow at the American Enterprise Institute and senior fellow in Jewish thought at the Shalem Center, and sat on the National Endowment for the Arts. He publishes widely; his work has appeared in "The Wall Street Journal", "New York Post", "Los Angeles Times", "The Weekly Standard", "Frankfurter Allgemeine Zeitung", and elsewhere. His paintings have been exhibited in New Haven and Manhattan.
  6. Title: Ann Arbor, Michigan. Current publications in the city include the Ann Arbor Journal (A2 Journal), a weekly community newspaper; the Ann Arbor Observer, a free monthly local magazine; the Ann Arbor Independent, a locally owned, independent weekly; and Current, a free entertainment-focused alt-weekly. The Ann Arbor Business Review covers local business in the area. Car and Driver magazine and Automobile Magazine are also based in Ann Arbor. The University of Michigan is served by many student publications, including the independent Michigan Daily student newspaper, which reports on local, state, and regional issues in addition to campus news.
  7. Title: New Haven, Connecticut. Livability.com named New Haven as the Best Foodie City in the country in 2014. There are 56 Zagat-rated restaurants in New Haven, the most in Connecticut and the third most in New England (after Boston and Cambridge). More than 120 restaurants are located within two blocks of the New Haven Green. The city is home to an eclectic mix of ethnic restaurants and small markets specializing in various foreign foods. Represented cuisines include Malaysian, Ethiopian, Spanish, Belgian, French, Greek, Latin American, Mexican, Italian, Thai, Chinese, Japanese, Vietnamese, Korean, Indian, Jamaican, Cuban, Peruvian, Syrian/Lebanese, and Turkish.
Rationale: The author of America-Lite: How Imperial Academia Dismantled Our Culture is David Gelernter. David Gelernter was educated at the Yale University. The city in Connecticut that has the highest number of Zagat-rated restaurants is New Haven. The weekly publication in New Haven that is issued by Yale University is Yale Herald.
Answer: Yale Herald

# Your Question to be answered
{content}

# Reference Context for Your Question
{context}

# Rationale Already Have
{rationale}

# Output Format
Your output should strictly follow the format below. Make sure your output parsable by json in Python.
{{
    "next_rationale": <The next sentence following to the rationale already have, ONLY one sentence.>,
    "answer": null
}}
or
{{
    "next_rationale": null,
    "answer": <Your answer to the given question>
}}

# Your Output
{limit}
""".strip())
    ],
    input_variables=["content", "context", "rationale", "limit"],
)


class IRCoTParser(BaseContentParser):
    def encode(
        self, content: str, rationales: List[str], references: List[str]=[], is_limit: bool=False, **kwargs,
    ) -> Tuple[str, Dict]:
        reference_strs = [f"  {i + 1}. {reference}" for i, reference in enumerate(references)]
        reference_str = "\n".join(reference_strs)
        return content, {
            "context": reference_str,
            "rationale": " ".join(rationales),
            "limit": "Null rationale allowed, output answer directly." if is_limit else "",
        }

    def decode(self, content: str, **kwargs) -> Dict[str, str]:
        try:
            output = parse_json(content)
        except Exception as e:
            print(f"[IRCoTParser] Content: {content}\nException: {e}")
            return {
                "next_rationale": None,
                "answer": None,
            }

        for key, value in output.items():
            if value is not None:
                output[key] = str(value)
        return output


ircot_qa_protocol = CommunicationProtocol(
    template=ircot_template,
    parser=IRCoTParser(),
)
