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


ircot_template_cn = MessageTemplate(
    template=[
        ("system", "你是一个擅长问答的有用AI助手。"),
        ("user", """
# 任务
您的任务是输出连续的推理句子或给定问题的最终答案。将提供四个演示，然后是要回答的问题和您可以参考的参考上下文。

# 演示
问题：Neville A. Stanton的雇主是什么时候成立的？
参考上下文：
  1. 标题：最后的马。最后的马（西班牙语：El último caballo）是1950年由Edgar Neville执导、Fernando Fernán Gómez主演的西班牙喜剧电影。
  2. 标题：南安普顿。南安普顿大学成立于1862年，1952年获得皇家大学特许状，拥有超过22,000名学生。该大学在2010年世界大学学术排名中排名世界前100所研究型大学。2010年，泰晤士高等教育-QS世界大学排名将南安普顿大学排在世界前80所大学之列。该大学认为自己是英国前5所研究型大学之一。该大学在工程科学、海洋学、化学、癌症科学、声音和振动研究、计算机科学和电子学、光电子学以及纺织品保护中心的纺织品保护方面享有全球声誉（该中心将于2009年10月关闭）。它也是国家海洋学中心南安普顿分部（NOCS）的所在地，这是自然环境研究委员会资助的海洋研究的焦点。
  3. 标题：伊利诺伊州香槟县斯坦顿镇。斯坦顿镇是美国伊利诺伊州香槟县的一个镇。截至2010年人口普查，其人口为505人，包含202个住房单元。
  4. 标题：Neville A. Stanton。Neville A. Stanton是南安普顿大学人因工程学和人体工程学的英国教授。Stanton教授是特许工程师（C.Eng）、特许心理学家（C.Psychol）和特许人体工程学家（C.ErgHF）。他撰写和编辑了超过四十本书和三百多篇同行评议期刊论文，涉及该学科的应用。Stanton是英国心理学会会员、人体工程学和人因学研究所会员以及工程技术学会成员。他曾在包括《自然》在内的学术期刊上发表文章。他还帮助组织设计新的人机界面，如捷豹汽车的自适应巡航控制系统。
  5. 标题：海底总动员。海底总动员院线发行海报 导演Andrew Stanton 制片人Graham Walters 编剧Andrew Stanton Bob Peterson David Reynolds 故事Andrew Stanton 主演Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe 配乐Thomas Newman 摄影Sharon Calahan Jeremy Lasky 剪辑David Ian Salter 制片公司华特迪士尼影片公司 皮克斯动画工作室 发行商博伟电影发行公司 上映日期2003年5月30日 片长100分钟 国家美国 语言英语 预算9400万美元 票房9.403亿美元
推理：Neville A. Stanton的雇主是南安普顿大学。南安普顿大学成立于1862年。
答案：1862

问题：制定ISO 21500标准的组织总部在哪里？
参考上下文：
  1. 标题：ISO 21500。ISO 21500:2012，项目管理指南，是国际标准化组织（ISO）从2007年开始制定并于2012年发布的国际标准。其目的是提供通用指导，解释核心原则以及项目管理的良好实践。处理项目管理的ISO技术委员会ISO/PC 236由美国国家标准协会（ANSI）主持，该协会批准了四个使用PMI材料的标准。其中之一是ANSI/PMI 99-001-2008，项目管理知识体系指南-第4版（PMI BoK®指南-第4版）（ANSI/PMI 99-001-2004的修订和重新指定）：2008年11月20日。
  2. 标题：ISO 3166-2:GH。ISO 3166-2:GH是ISO 3166-2中加纳的条目，ISO 3166-2是国际标准化组织（ISO）发布的ISO 3166标准的一部分，它定义了ISO 3166-1中编码的所有国家的主要分区（例如省或州）名称的代码。
  3. 标题：ISO 4031。ISO 4031是国际标准化组织于1978年首次发布的国际标准。它定义了本地时间差的表示，通常称为时区。它此后被较新的标准ISO 8601取代。这个较新的标准规定了本地时间差的当前格式，因此ISO 4031不再使用。
  4. 标题：ISO/TC 68。ISO/TC 68是在瑞士日内瓦的国际标准化组织（ISO）内成立的技术委员会，负责制定和维护涵盖银行、证券和其他金融服务领域的国际标准。作为ISO下负责制定所有国际金融服务标准的标准组织，ISO/TC 68在银行、经纪和保险行业新技术的发展和采用中发挥着关键作用。其当前的许多工作项目涉及制定电子商务标准，如金融交易的更好在线安全性、金融交易的XML标准以及减少国际金融交易成本和延迟的标准。ISO/TC 68的成员包括参与国家标准机构指定的30多个组织，以及与全球金融服务标准开发协作工作的其他国际标准开发组织。
  5. 标题：ISO 3166-1。ISO 3166-1是国际标准化组织（ISO）发布的ISO 3166标准的一部分，定义了国家、依赖领土和地理利益特殊区域名称的代码。该标准的正式名称是"国家及其分区名称表示代码-第1部分：国家代码"。它定义了三套国家代码：
推理：ISO 21500的标准由国际标准化组织制定。国际标准化组织的总部在日内瓦。
答案：日内瓦

问题：Smoke in tha City表演者的出生地在哪个县？
参考上下文：
  1. 标题：阿肯色州切罗基城。切罗基城是美国阿肯色州本顿县的一个非建制人口普查指定地区。截至2010年人口普查，其人口为72人。它是库恩溪桥的所在地（或最近的社区），该桥位于县道24号上，并被列入国家历史名胜名录。该社区以切罗基印第安人命名，因为当切罗基人在1830年代后期向西迁移到现在的俄克拉荷马州印第安领土时，眼泪之路穿过了这片土地。该镇距离俄克拉荷马州约5英里，距离密苏里州州界以南4英里。
  2. 标题：加利福尼亚州康普顿。康普顿是美国加利福尼亚州洛杉矶县南部的一个城市，位于洛杉矶市中心以南。康普顿是该县最古老的城市之一，1888年5月11日是第八个合并的城市。截至2010年美国人口普查，该市总人口为96,456人。由于其在洛杉矶县的地理中心位置，它被称为"枢纽城市"。康普顿的社区包括阳光湾、利兰、康普顿市中心和里奇兰农场。该市一般是工人阶级城市，有一些中产阶级社区，人口相对年轻，平均年龄为25岁，而美国中位年龄为38岁（基于2018年数据）。
  3. 标题：MC Eiht。Aaron Tyler（1971年5月22日出生），以其艺名MC Eiht更为人知，是美国说唱歌手和演员。他的许多歌曲都基于他在康普顿的生活。他的艺名部分受到KRS-One名字中数字的启发。他选择Eiht是因为它与"街区文化"的联系，包括老英语800（8球）和.38口径枪支。他是西海岸嘻哈组合康普顿最想要的"事实上"领袖，该组合还包括康普顿籍说唱歌手Boom Bam、Tha Chill、DJ Mike T、DJ Slip和Ant Capone。他还以在1993年电影《威胁II社会》中饰演A-Wax而闻名。
  4. 标题：Smoke in tha City。Smoke in tha City是美国说唱歌手MC Eiht的第九张录音室专辑，2004年9月14日由Factor House Records发行。由Black C-Zer和Quincy Miller制作。专辑特邀西海岸重量级客串：RBX、Spice 1、Kokane、Jayo Felony和Daz Dillinger。
  5. 标题：碧昂丝。2012年1月7日，碧昂丝在纽约勒诺克斯山医院生下了她的第一个孩子，女儿Blue Ivy Carter。五个月后，她在Revel大西洋城的Ovation Hall演出四晚以庆祝度假村开业，这是她生下Blue Ivy后的首次演出。
  6. 标题：奥尔什丁省。奥尔什丁省是波兰在1945-75年间的行政区划和地方政府单位，1975-1998年间的新领土区划，被瓦尔米亚-马祖里省取代。其首府是奥尔什丁。
推理：Smoke in tha City的表演者是MC Eiht。MC Eiht的出生地是康普顿。康普顿位于洛杉矶县。
答案：洛杉矶县

问题：在康涅狄格州拥有最多Zagat评级餐厅的城市中，由《America-Lite: How Imperial Academia Dismantled Our Culture》作者所在大学发行的周刊是什么？
参考上下文：
  1. 标题：康涅狄格州纽黑文。纽黑文由日报《纽黑文纪事报》、周刊"另类"《纽黑文倡导者》（由拥有《哈特福德新闻报》的Tribune公司运营）、在线日报《纽黑文独立报》和月刊《大新闻社区报》服务。纽黑文市中心由深度公民新闻论坛Design New Haven覆盖。《纪事报》还支持PLAY杂志，一份周刊娱乐出版物。该市还由几份学生办报纸服务，包括《耶鲁日报》、周刊《耶鲁先驱报》和幽默小报《Rumpus杂志》。WTNH第8频道，康涅狄格州的ABC附属电视台，WCTX第59频道，该州的MyNetworkTV附属电视台，以及康涅狄格公共电视台WEDY第65频道，PBS附属电视台，都从纽黑文播出。所有纽约市新闻和体育团队电视台都向纽黑文县播出。
  2. 标题：伦敦帝国理工学院。帝国理工学院联盟，帝国理工学院的学生会，由从学生团体中选出的五名全职sabbatical官员管理，任期一年，以及一些常设工作人员。联盟从大学获得大量津贴，其中大部分用于维护约300个俱乐部、项目和社团。著名学生团体和项目的例子包括尼泊尔项目，该项目派遣帝国理工学院学生到尼泊尔农村参与教育发展计划，以及萨尔瓦多项目，这是中美洲的一个建设项目。联盟还主办体育相关俱乐部，如帝国理工学院赛艇俱乐部和帝国理工学院滑翔俱乐部。
  3. 标题：教育的终结。《教育的终结》是Neil Postman关于美国公共教育的一本书。标题中"终结"一词有两个含义：主要是作为"目的"的同义词，但也是对公立学校未来的预测，如果它们不能成功地识别和传达其在我们文化中存在的令人信服的目的。
  4. 标题：America-Lite。America-Lite: How Imperial Academia Dismantled Our Culture (and Ushered in the Obamacrats)是David Gelernter 2012年由Encounter Books出版的一本书。
  5. 标题：David Gelernter。David Hillel Gelernter（1955年3月5日出生）是美国艺术家、作家和耶鲁大学计算机科学教授。他曾是美国企业研究所国家研究员和沙勒姆中心犹太思想高级研究员，并曾任国家艺术基金会委员。他发表了大量作品；他的作品出现在《华尔街日报》、《纽约邮报》、《洛杉矶时报》、《每周标准》、《法兰克福汇报》等地方。他的画作曾在纽黑文和曼哈顿展出。
  6. 标题：密歇根州安娜堡。该市目前的出版物包括《安娜堡日报》（A2日报），一份周刊社区报纸；《安娜堡观察家》，一份免费的月刊地方杂志；《安娜堡独立报》，一份当地拥有的独立周刊；以及《Current》，一份免费的娱乐导向另类周刊。《安娜堡商业评论》覆盖该地区的本地商业。《汽车与驾驶员》杂志和《汽车》杂志也设在安娜堡。密歇根大学由许多学生出版物服务，包括独立的《密歇根日报》学生报纸，除了校园新闻外，还报道地方、州和地区问题。
  7. 标题：康涅狄格州纽黑文。Livability.com在2014年将纽黑文评为全国最佳美食城市。纽黑文有56家Zagat评级餐厅，是康涅狄格州最多的，也是新英格兰第三多的（仅次于波士顿和剑桥）。超过120家餐厅位于纽黑文绿地两个街区内。该市拥有各种民族餐厅和专门经营各种外国食品的小市场。代表的菜系包括马来西亚、埃塞俄比亚、西班牙、比利时、法国、希腊、拉丁美洲、墨西哥、意大利、泰国、中国、日本、越南、韩国、印度、牙买加、古巴、秘鲁、叙利亚/黎巴嫩和土耳其菜。
推理：《America-Lite: How Imperial Academia Dismantled Our Culture》的作者是David Gelernter。David Gelernter在耶鲁大学接受教育。康涅狄格州拥有最多Zagat评级餐厅的城市是纽黑文。纽黑文由耶鲁大学发行的周刊是《耶鲁先驱报》。
答案：耶鲁先驱报

# 您要回答的问题
{content}

# 您问题的参考上下文
{context}

# 已有的推理
{rationale}

# 输出格式
您的输出应严格遵循以下格式。确保您的输出可以被Python中的json解析。
{{
    "next_rationale": <跟随已有推理的下一句话，只有一句话。>,
    "answer": null
}}
或
{{
    "next_rationale": null,
    "answer": <您对给定问题的答案>
}}

# 您的输出
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


ircot_qa_protocol_cn = CommunicationProtocol(
    template=ircot_template_cn,
    parser=IRCoTParser(),
)
