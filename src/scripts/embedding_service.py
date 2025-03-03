from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone.core.openapi.inference.models import EmbeddingsList
import os
from dotenv import load_dotenv, find_dotenv

class PineconeEmbeddingManager:
    def __init__(self, api_key: str, index_name: str, name_space: str):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.name_space = name_space
    
    def create_embeddings(self, documents: List[Document], model_name: str = "llama-text-embed-v2") -> EmbeddingsList:
        return self.pc.inference.embed(
            model=model_name,
            inputs=[d.page_content for d in documents],
            parameters={"input_type": "passage", "truncate": "END"}
        )
    
    def store_embeddings(self, embeddings: EmbeddingsList, documents: List[Document]):
        index = self.pc.Index(name=self.index_name)
        records = [
            {"id": f"vector{idx}", "values": e['values'], "metadata": {"text": d.page_content}}
            for idx, (d, e) in enumerate(zip(documents, embeddings))
        ]
        return index.upsert(vectors=records, namespace=self.name_space)
    
    def create_and_store_embeddings(self, documents: List[Document]):
        embeddings = self.create_embeddings(documents)
        result = self.store_embeddings(embeddings, documents)
        print(result)
    
    def search_matching(self, query: str, model: str = "llama-text-embed-v2", top_k: int = 3):
        index = self.pc.Index(name=self.index_name)
        query_embedding = self.pc.inference.embed(
            model=model,
            inputs=[query],
            parameters={"input_type": "query"}
        )


        results =  index.query(
            namespace=self.name_space,
            vector=query_embedding[0].values,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )['matches']

        documents = [result['metadata']['text'] for result in results]

        return documents
        
if __name__ == '__main__':
    load_dotenv(find_dotenv())
    api_key = os.environ.get('PINECONE_API_KEY')
    manager = PineconeEmbeddingManager(api_key=api_key, index_name='kifiya', name_space='test')
    
    markdown_document = '''### **About Kifiya**  

**Pioneers of Digital Financial Technology Excellence**    
Crafting a financially and digitally inclusive future for our continent since 2010\.  

\- **9+** Tech platforms    
\- **550+** Employees    
\- **20+** Regions    
\- **25+** Languages  

### **Who We Are**  

**Vision**    
Enabling equitable access to financial services and markets for Africa.  

**Mission**    
Delivering data-driven, intelligent financial and market solutions; making them within reach for all.  

### **What We Do**  

We construct state-of-the-art digital infrastructure for credit, insurance, and payments, empowering banks and non-bank financial institutions to provide uncollateralized credit to MSMEs and consumers at scale. We offer a marketplace connecting MSMEs with larger businesses and consumers. Our solutions digitize agricultural value chains, connecting smallholder farmers with improved input, credit, and larger markets. We also streamline booking and ticketing systems for both short and long-distance travel.  

Our platform is built to deliver around four areas:  

1\. **For Financial Institutions**    
   We deliver state-of-the-art digital infrastructure for credit, insurance, and payments, empowering banks and non-bank financial institutions to provide uncollateralized credit to MSMEs and consumers at scale.  

2\. **For MSMEs**    
   Our multi-sided digital marketplace connects them with larger businesses and consumers, enabling them to access markets and scale their businesses.  

3\. **For the Agriculture Value Chain**    
   Our digitized solution connects smallholder farmers with improved input, credit, and larger markets.  

4\. **For Transport**    
   We streamline booking and ticketing systems for both short and long-distance travel.  

### **Our History**  

Over the last 14 years, Kifiya has transformed from a startup into a leading innovator in digital financial services and AI-based modeling, advancing access to financial services and markets in Ethiopia. This journey has vividly demonstrated the immense potential of big data, not just in our capacity to harness it but also to innovate upon it. Our dedication to enhancing our platform, designing inclusive products, empowering our team, and advancing our technology remains unwavering, driving us toward even greater achievements.  

\- **2012**    
  We embarked on our mission as a payments company, pioneering the unification and digitization of national utility bill payments through a Build-Operate-Transfer model. In collaboration with the Gates Foundation, we transformed the Productive Safety Net Program (PSNP) distribution from cash to electronic payments, a groundbreaking public-private partnership (PPP) chosen by McKinsey. This marked the beginning of our journey to redefine financial transactions in Ethiopia.  

\- **2015**    
  Our journey took a bold turn towards diversification as we ventured into agricultural insurance, solar energy, and airtime distribution. These initiatives underscored our dedication to leveraging technology for societal benefit, addressing critical needs in agriculture, energy, and communication sectors.  

\- **2018**    
  With the digital landscape evolving rapidly, we expanded our horizons into e-commerce, mobility, and digital finance. This expansion was driven by our vision to offer comprehensive digital solutions, facilitating easier, more efficient transactions and services for the Ethiopian people.  

\- **2022**    
  Embracing the era of artificial intelligence, we transformed into a venture-building company, innovating in the realm of AI for credit scoring. This leap forward represents our commitment to financial inclusivity, using cutting-edge technology to democratize access to credit and financial services.  

\- **2024**    
  In 2024, Kifiya has evolved into a product-led tech company, significantly scaling our offerings across Ethiopia and the digital financial industry. We are actively working with six banks and significant FMCG players, with nine products ready to be rolled out into the market. Additionally, we have achieved an international team composition and are eyeing and exploring expansion across Sub-Saharan Africa, further cementing our role as a regional leader in innovation and development.  

### **Who We Serve**  

We serve a wide array of customers and clients that support our innovative products and collaborate in the delivery of our solutions.  

#### **1\. Financial Institutions (FIs) and Non-Bank Financial Institutions (NBFIs)**  

**Unlock Digital Finance Innovation with Kifiya’s Cutting-edge Solutions**    
Our innovative solutions are tailored for Financial Institutions and Non-Bank Financial Institutions looking to simplify & secure access to financial services. We are reshaping the financial landscape by providing state-of-the-art digital infrastructure. From AI-enabled credit assessments to seamless payment processes, we empower FIs and NBFIs to transcend traditional limitations and harness the power of fintech innovation.  

**How We Serve**    
\- **Agency Banking & Origination**    
  Effortlessly expand your customer base with our comprehensive agency banking solutions. We facilitate customer acquisition and extend banking services across vast networks, ensuring financial inclusivity and reach.    
\- **Digital Credit and Payment Infrastructure**    
  We provide state-of-the-art digital infrastructure, from AI-enabled credit assessments to seamless payment processes.    
\- **Embedded Finance Solutions**    
  Our B2B and B2C platforms seamlessly integrate finance into commerce, enabling businesses to thrive in the digital economy with tools like Buy Now, Pay Later, and inventory credit.  

**Why Choose Kifiya?**    
\- **Tailored Fintech Ecosystem**    
  Embrace a suite of digital products designed to meet the dynamic needs of today’s financial services landscape. From credit scoring to digital payments, our platforms are customizable for your institution’s requirements.    
\- **Innovative Partnerships**    
  Leverage our established collaborations with telecoms, payment partners, and government sectors, adding layers of capability and service to your offerings.    
\- **Dedicated Support and Training**    
  Benefit from our market expertise with access to continuous support, technical assistance, and skill development programs to ensure smooth adoption and maximization of our fintech solutions.    
\- **Real Success Stories**    
  Hear from our partners who have witnessed transformative growth by integrating our digital finance solutions. Discover how we’ve helped institutions overcome challenges, enhance operational efficiency, and unlock new market opportunities.  

#### **2\. MSMEs and Consumers**  

**Bridging The Gap Between Financial Opportunities and The Communities We Serve**    
Kifiya is dedicated to universalizing access to financial services, ensuring that Micro, Small, and Medium Enterprises (MSMEs), alongside consumers, can easily navigate the digital economy. Our comprehensive solutions bridge the gap between financial opportunities and the communities we serve.  

**How We Serve**    
\- **Embedded Finance for Seamless Transactions**    
  Integrate financial services directly into your business model with Kifiya’s Embedded Finance. This includes inventory credit and innovative payment solutions that empower MSMEs to grow and serve consumers better.    
\- **Digital Credit and E-commerce Platforms**    
  Access uncollateralized loans and engage in digital marketplaces with ease. Our B2B|B2C platforms support MSMEs in manufacturing, agri-food systems, and service sectors, facilitating market access and enhancing operational efficiency.    
\- **MicroInsurance with a Difference**    
  Kifiya’s InsurTech empowers MSMEs and consumers with Health and Edir Micro Insurance, providing tailored financial protection through an innovative marketplace that connects them with a range of insurance products. Leveraging data-driven risk modeling and a strong network of partnerships, we extend the reach of microinsurance to hundreds of thousands, including a significant focus on women and small businesses, fostering economic resilience and contributing to a more secure financial future for our community.    
\- **Agency Banking for Wider Reach**    
  Extend the reach of financial services to underserved areas. Kifiya’s agency banking model ensures that both MSMEs and consumers can access banking services, even in remote locations.  

**Why Partner with Kifiya**    
\- **Inclusive Financial Ecosystem**    
  Join an ecosystem that’s built on the foundation of financial inclusivity. Our solutions cater to the unique needs of MSMEs and consumers, ensuring that financial barriers are no longer an obstacle to economic participation and growth.    
\- **Customizable Solutions**    
  Leverage Kifiya’s flexible platforms that adapt to your specific needs. Whether it’s accessing credit, engaging with e-commerce, or utilizing payment services, we have a solution for you.    
\- **Supporting Community Development**    
  Beyond financial services, Kifiya is committed to the broader economic development of our communities. Our solutions are aimed at not just financial success but also at fostering sustainable growth for MSMEs and improving the livelihoods of consumers.    
\- **Success Stories**    
  Discover how Kifiya has transformed the financial landscape for MSMEs and consumers. From facilitating access to credit for small businesses to simplifying payments for consumers, our impact stories highlight the real-world benefits of our solutions.  

#### **3\. Smallholder Farmers**  

**Empowering Smallholder Farmers with Digital Solutions**    
At Kifiya, we aim to transform the agricultural ecosystem through digital innovation, providing smallholder farmers with the tools they need to thrive. Our technology-driven solutions are designed to increase access to financial services, markets, and essential agricultural inputs, empowering farmers towards greater productivity and sustainability.  

**Our Solutions**    
\- **AgriTech Platforms**    
  Leveraging digital technology to connect smallholder farmers with vital resources. Our platforms offer access to improved inputs, credit, and information on sustainable farming practices, all aimed at enhancing crop yields and resilience against climate variability.    
\- **eVoucher Systems**    
  Simplifying the distribution of agricultural subsidies and inputs through secure, digital vouchers. This system ensures that farmers have timely access to seeds, fertilizers, and other inputs necessary for successful cultivation.    
\- **Agricultural Marketplaces**    
  Digital marketplaces that provide smallholder farmers with direct access to buyers, enabling fairer pricing and expanding market reach. This reduces the dependency on intermediaries, ensuring more of the profit goes directly to the farmers.    
\- **Microinsurance for SHFs**    
  Through our Agri InsurTech, we harness the power of Normalized Difference Vegetation Index (NDVI) data to offer innovative parametric and yield insurance, safeguarding smallholder farmers against the uncertainties of climate and crop yield risks.  

**Impact and Success Stories**    
Our interventions have led to remarkable outcomes for smallholder farmers across regions. Success stories include:    
\- Increased agricultural productivity and income through access to credit and high-quality inputs.    
\- Enhanced market access leading to better pricing and reduced post-harvest losses.    
\- Empowerment of farmer groups through collective selling and purchasing.  

**Aligning with Global Goals**    
Kifiya’s efforts with smallholder farmers directly contribute to the United Nations Sustainable Development Goals, particularly:    
\- **Goal 2:** Zero Hunger by improving agricultural productivity.    
\- **Goal 8:** Decent Work and Economic Growth by increasing income opportunities.    
\- **Goal 12:** Responsible Consumption and Production by promoting sustainable agriculture.  

#### **4\. Commuters**  

**Transform Your Travel with Kifiya: Seamless Commuting Reimagined**    
Navigate the complexities of daily transport with ease thanks to Kifiya’s digital commuting solution. Our platform offers affordable, reliable, and convenient travel experiences and financial solutions that make managing travel costs straightforward and accessible.  

**How We Serve**    
\- **Booking Made Easy**    
  Kifiya’s digital platforms allow users to book their bus journey effortlessly, whether it’s for short commutes within the city or longer trips to the countryside. They can enjoy the simplicity of managing their travel plans from their mobile device, with all the information they need at their fingertips.    
\- **Dependable and Fair**    
  Our system ensures that buses are where they need to be when they need to be, providing a dependable service that commuters can trust. With transparent pricing and no hidden fees, users can plan their budget and route with confidence.    
\- **Commitment to Sustainability**    
  Digital ticketing means less paper waste, and our route optimization contributes to lower emissions — making the commute not only smart but also environmentally conscious. With the future induction of electronic vehicles, our commitment to sustainability will be further affirmed.    
\- **Secure and Safe Journeys**    
  Our digital ticketing platform offers secure payment options and keeps personal information protected.  

### **Our Ventures**  

Innovative products that revolutionize:    
\- Market Systems    
\- Agriculture    
\- Financial Services    
\- Last-mile Delivery  

#### **1\. Expanding Financial Accessibility** **Intelligent Financial Services (IFS) Platform**  

Unlock the full potential of your financial services with Kifiya's Agency Banking and Origination solutions. Designed to bridge the accessibility gap in financial services, our platform empowers financial institutions to extend their reach into underserved communities, making financial products more accessible and convenient.  

**Service Overview**    
Kifiya’s Agency Banking service enables financial institutions to operate through a network of agents, thereby extending banking services to remote areas where traditional bank branches are not viable. Our Origination service complements this by providing a streamlined process for acquiring new customers, leveraging technology to simplify and speed up the registration and application processes for financial products.  

**Benefits**    
1\. **Agent Network Setup**    
   Access remote and underserved markets with minimal overhead, increasing your customer base.    
2\. **Operational Efficiency**    
   Reduce the cost and complexity of serving new and existing customers.    
3\. **Enhanced Customer Experience**    
   Offer your customers convenient access to financial services, improving satisfaction and loyalty.  

**Success Stories**    
From rural outreach programs that have connected thousands of smallholder farmers to banking services, to urban initiatives that have simplified banking for busy city dwellers, our partners have seen tangible benefits from implementing Agency Banking and Origination solutions. These stories underscore the transformative potential of digital finance when strategically deployed.  

#### **2\. Digital Credit and Payment Infrastructure**  

Kifiya is pioneering the future of financial transactions with our Digital Credit and Payment Infrastructure, designed to bridge the gap between financial services and those who need them most. Our innovative platforms empower financial institutions, MSMEs, and consumers with more accessible, secure, and efficient financial transactions.  

**01\. Advancing Financial Access with Digital Solutions**    
Kifiya is pioneering the future of financial transactions with our Digital Credit and Payment Infrastructure, designed to bridge the gap between financial services and those who need them most. Our innovative platforms empower financial institutions, MSMEs, and consumers with more accessible, secure, and efficient financial transactions.  

**02\. Digital Credit Services**    
At the heart of our venture is the commitment to provide inclusive financial products. Our digital credit solutions harness advanced analytics and alternative credit scoring to offer uncollateralized loans, making financial support accessible to underserved markets. This initiative is about more than just loans; it’s about fostering financial inclusion and enabling growth.  

**03\. Seamless Payment Infrastructure**    
Kifiya’s payment infrastructure revolutionizes how transactions are conducted, from utility payments to complex financial operations. Our platform ensures transactions are not just seamless but also secure, catering to a wide array of payment needs across different sectors. Whether it’s for individual consumers or for business transactions, Kifiya simplifies payments, making them more accessible and reliable.  

**04\. Impact Through Innovation**    
Our success stories highlight the transformative effect of Kifiya’s digital credit and payment solutions. From enabling small businesses to thrive through accessible credit options to simplifying daily transactions for millions of consumers, our technology is at the forefront of financial innovation.  

**05\. Integration Made Easy**    
Understanding the technical complexities of integration, Kifiya offers comprehensive support to partners. With detailed documentation, APIs, and dedicated technical assistance & change management, we ensure that integrating our digital credit and payment solutions into your services is straightforward, enhancing your offerings and extending your reach.  

#### **3\. Embedded Finance**  

Chegebeya is a super platform that enables eCommerce (B2B and B2C) and integrates embedded finance, i.e., uncollateralized inventory & consumer loans (BNPL), plus other products like payments & insurance, making financial services frictionless and convenient for retailers and consumers.  

**Our Solutions**    
\- **B2B**    
  Chegebeya B2B eCommerce provides digital rails to value chain players, i.e., manufacturers, distributors, and retailers. They can make orders and pay via supply chain financing at check-out. Suppliers are paid in real-time, and retailers can pay back the loan as they sell the goods.    
\- **B2C**    
  Chegebeya enables vendors/SMEs to create online shops within our eComm site, benefiting from enhanced marketing and consumer traffic locally and globally. Consumers shop, buy, and pay later via our BNPL option or check-out via various integrated digital payments. Our eComm plug-in, Che BNPL check-out, also enables consumers to buy and pay later via various other websites of their choice.    
\- **White Label Lending APIs**    
  For brand owners and distribution companies that have their own systems, we partner to integrate lending APIs so that they can extend credit to their customers.  

#### **4\. Agtech**  

**Empowering Smallholder Farmers for Sustainable Development**    
Kifiya leverages cutting-edge AgriTech solutions to support Smallholder Farmers (SHFs), driving sustainable agricultural practices and economic growth. Our digital innovations offer scalable solutions for development organizations, international NGOs, government entities, and local cooperatives, aiming to transform the agricultural landscape and enhance food security.  

Our comprehensive AgriTech suite is designed to address critical challenges in agriculture, fostering resilience and productivity among SHFs.  

**01\. eVoucher Systems**    
Streamlines the distribution of agricultural inputs through digital vouchers, ensuring timely access to essential resources, supported by transparent tracking for funders and stakeholders.  

**02\. Digital Marketplaces**    
Facilitates direct connections between SHFs and markets, improving income opportunities and reducing post-harvest losses, while offering data insights for market trends analysis.  

**03\. Remote Sensing Technologies**    
Employs advanced satellite imagery to provide actionable agricultural insights, supporting decision-making for optimized crop management and environmental sustainability.  

**04\. Agri MicroInsurance**    
Empowers smallholder farmers by offering tailored, affordable insurance products, protecting them against the unpredictability of weather and crop yields. This crucial support mitigates risk, ensuring farmers can sustain their livelihoods and contribute to food security with confidence.  

**Partnership Approach**    
Through partnerships with leading financial and agricultural institutions, we have significantly advanced the credit and insurance landscape. From enabling thousands of MSMEs to access uncollateralized loans to securing crop insurance for smallholder farmers, our data science solutions create real-world impact.

###  **Market Solutions Practice**  

#### **Delivering Impact**  

At Kifiya, the Market Solutions Practice (MSP) team is dedicated to addressing market failures and catalyzing significant social impact by designing and implementing innovative solutions in collaboration with partners.  

#### **Key Impact-Driven Initiatives Managed by MSP**  

**01\. SAFEE (Sustainable Access to Finance for Enabling Entrepreneurship)**    
Unlocking financial opportunities for over **477,861 micro, small, and medium enterprises (MSMEs)** by providing uncollateralized digital credit products. The program also supports **425,000 young women** through mobile device financing, aiming to create work opportunities for up to **2.18 million youth** over five years.    
\- **Contributing to SDG 5:** Gender Equality    
\- **Contributing to SDG 8:** Decent Work and Economic Growth    
\- **Contributing to SDG 9:** Industry, Innovation & Infrastructure  

**02\. UNCDF Women Digital Literacy Program**    
Closing the digital gender gap, this initiative empowers women with digital financial skills through a **5-month training program**, enhancing their participation in digital financial services.    
\- **Contributing to SDG 5:** Gender Equality  

**03\. UNCDF, Access to Market Program**    
Facilitating supply chain credit for retailers and digitizing traditional markets to support financially excluded merchants.    
\- **Contributing to SDG 8:** Decent Work and Economic Growth  

**04\. Mastercard E-commerce Marketplace**    
Bridging the finance gap for micro, small, and medium enterprises (MSMEs) by providing access to broader markets through digital e-commerce tools.    
\- **Contributing to SDG 1:** No Poverty    
\- **Contributing to SDG 8:** Decent Work and Economic Growth    
\- **Contributing to SDG 9:** Industry, Innovation & Infrastructure  

**05\. KFW & ISF Partnership**    
Enhancing the climate resilience of smallholder farmers by providing tailored insurance products. The program utilizes satellite vegetation index technology to measure the extent of climate impact.    
\- **Contributing to SDG 13:** Climate Action  

#### **Our Impact By the Numbers**  

\- Our first uncollateralized digital credit was provided through our BPAAS product with one bank.    
\- **35%** Women    
\- **68%** Youth    
\- **18%** Enterprises employ vulnerable groups    
\- **$57 million** MSMEs accessed in financing    
\- **$158K** Worth of vouchers were issued through cash or credit    
\- **2.43 million** MSMEs gained market access    
\- **14,280** Micro retailers accessed supplies    
\- **768,000** Farmers repaid debts totaling over ETB 52 million  

#### **SAFEE (Sustainable Access to Finance for Enabling Entrepreneurship)**  

The SAFEE program aims to transform access to finance for women- and youth-led enterprises, driving financial inclusion and creating job opportunities for young people. With a **US$ 75 million transitional liquidity and guarantee facility**, along with a **US$ 25 million technical assistance facility**, SAFEE deploys innovative, data-driven credit scoring technology solutions to unlock up to **US$ 300 million** from six banks. This will enable more than **477,861 micro, small, and medium enterprises (MSMEs)** to access uncollateralized digital credit products, while providing **425,000 young women** with access to mobile device financing. Over the next five years, the program is expected to create up to **2.18 million work opportunities for youth** and support other initiatives to generate an additional **3.79 million opportunities**.  

##### **Target Participants**  

SAFEE supports a diverse range of participants, ensuring that its financial solutions reach those most in need. Participants are categorized based on geographical location and gender inclusivity. The SAFEE program focuses on **rural, urban, and peri-urban areas**, providing tailored financial solutions that address the unique needs of these communities. Businesses must meet essential social, environmental, and governance criteria, excluding sectors such as alcohol, tobacco, weapons manufacturing, gambling, and state ownership.  

1\. **Women**    
   \- Women-owned enterprises (**80%+**) are prioritized.    
   \- Enterprises managed by women, employing a majority of women, or with a majority of women members.  

2\. **Youth**    
   \- Youth-owned businesses (**80%+**) are prioritized.  

3\. **Persons Living With Disabilities**    
   \- Enterprises owned by or employing a majority of persons with disabilities (**10%**).  

4\. **Refugees & IDPs**    
   \- Enterprises owned by refugees, IDPs, or returnees (**10%**).    
   \- Enterprises employing refugees, IDPs, or returnees.  

##### **Banks & FinTech**  

SAFEE collaborates with the following banking partners to drive financial inclusion:    
\- Cooperative Bank of Oromia    
\- Amhara Bank    
\- Bunna Bank    
\- Enat Bank    
\- Wegagen Bank    
\- ZamZam Bank  

Future collaborations with fintechs and microfinance institutions (MFIs) will be included as they are finalized.  

##### **Enabling Environment**  

SAFEE works within a robust ecosystem to promote financial inclusion, collaborating with key stakeholders such as:    
\- Academic Institutions (local and international)    
\- National Bank of Ethiopia (NBE)    
\- Mobile Network Operators    
\- Financial Sector Stakeholders    
\- Ministry of Labor and Social Affairs (MoLS)  

Additional collaborations will continue to expand as the program evolves.  

###  **AI Lab**  

**Harnessing Data to Unlock Financial Opportunities**    
In a world where data is the new currency, Kifiya's Data Science venture stands at the cutting edge, transforming vast amounts of information into actionable insights that drive financial inclusion and market accessibility.  

#### **Empowering Decisions with Advanced Analytics**  

**01\. Credit Scoring Models**    
We develop AI-driven scoring models that give financial institutions the tools to extend credit to underserved markets confidently. Our proprietary algorithms consider a broad spectrum of data points, providing a holistic view of creditworthiness beyond traditional metrics.  

**02\. Innovative Risk Assessment**    
Our data science goes beyond credit, delving into insurance risk with remote sensing data that evaluates the health of crops for smallholder farmers, ensuring they receive loans and insurance tailored to their needs.  

**03\. Building for the Future**    
Kifiya’s data science journey is about anticipating the needs of tomorrow. Our ventures into alternative credit scoring and predictive modeling are paving the way for a more inclusive financial landscape, where data-driven decisions help to democratize access to finance for all.  

**04\. Driving Change Through Data**    
\- **Real-time Data Processing:** Our state-of-the-art infrastructure captures and processes real-time data, ensuring our partners always have the most current and relevant information at their fingertips.    
\- **Predictive Analytics:** By leveraging machine learning, we’re not just reacting to market trends — we’re staying ahead of them, equipping our partners with predictive insights that inform smarter, more strategic decision-making.  

**05\. Predictive Analytics**    
By leveraging machine learning, we’re not just reacting to market trends — we’re staying ahead of them, equipping our partners with predictive insights that inform smarter, more strategic decision-making.  

#### **Our Impact in Action**  

Through partnerships with leading financial and agricultural institutions, we have significantly advanced the credit and insurance landscape. From enabling thousands of MSMEs to access uncollateralized loans to securing crop insurance for smallholder farmers, our data science solutions create real-world impact.  

#### **Collaborate with Kifiya's Data Science Team**  

Discover the potential of data science to revolutionize your operations and decision-making processes. Get in touch with us today to explore Kifiya’s data-driven solutions and services.  

### **Job Board**  

**Disclaimer**    
Please remain vigilant against fraudulent job postings claiming to be from Kifiya on external websites. All official Kifiya job openings are exclusively listed on our careers site. Kifiya strictly adheres to never conducting interviews via chat or text, nor do we request any upfront payments or deposits from candidates. Should you encounter any suspicious activity, please promptly contact us at \*\*hiring@kifiya.com\*\* with any pertinent information you have.  

Kifiya takes pride in being an equal opportunity employer and is committed to affirmative action. We ensure equal employment opportunities for all applicants and current employees, evaluating qualifications without discrimination based on race, creed, color, national origin, sex (including pregnancy and gender identity/expression), sexual orientation, age, ancestry, physical or mental disability, marital status, political affiliation, religion, citizenship status, genetic information, or any other protected characteristic under relevant local laws.  

#### **Find Jobs Here**  

Explore exciting career opportunities with Kifiya and join a team that is transforming digital financial technology across Africa. Visit our official job board to view all current openings and apply directly:    
**https://kifiya.com/job-board/**

At Kifiya, we value innovation, inclusivity, and collaboration. If you’re passionate about driving financial inclusion and leveraging cutting-edge technology to make a difference, we’d love to hear from you.  

### **Contacts**  

#### **Get in Touch**  

For general inquiries, feel free to reach out to us via email or phone. We’d love to hear from you\!  

\- **Email:** info@kifiya.com  
\- **Phone:** \+251 116 671 579  

#### **Follow Us on Social Media**  

Stay connected with Kifiya and keep up with the latest updates, news, and opportunities:  

\- **Facebook:** https://www.facebook.com/KifiyaFT/  
\- **X (formerly Twitter):**https://twitter.com/kifiyaft?lang=en    
\- **LinkedIn:** https://www.linkedin.com/company/kifiya-financial-technology-plc/?originalSubdomain=et  
'''

    headers_to_split_on = [("###", "Header 3"), ("####", "Header 4")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_document)
    
    manager.create_and_store_embeddings(md_header_splits, index_name='kifiya', name_space='test')
    
    result = manager.search_matching("Where can I contact kifiya?")
    for doc in result:
        print(doc)
