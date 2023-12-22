# Data Collection

This step is crucial and will ultimately decide the success, or failure of this project. Despite the increasigly sophisticated language models of our day, when it comes to machine learning there is one rule above the rest:  Garbage in, garbage out. For this reason, the data collected and ultimately generated for this project needs to be contextually relevant, and deliver good examples for the models to use from the perspective of prospective students, current students, and faculty alike.

The primary source of data at our disposal comes directly from the SUNY Brockport website (https://www2.brockport.edu). It may be possible, if necessary, to request more information from the relevant people, but this should be adequate. 

**NOTE:** *Files in this directory are labeled 1,2,3, etc. to indicate the order in which they should be run since they are dependent on each other.*

---
## Scraping Brockport Website

Using data from the website requires us to scrape the site. I've implemented a version of this in [brockport_scraping.py](./brockport_scraping.py). In that file, data is collected recursively. From some set of links (or one, in my case, the [SUNY Brockport Homepage](https://www2.brockport.edu/)) all their webpages are retreived. From there I collect all the links on each of those webpages, and apply a set of filters. Most importantly, these filters include: obeying robots.txt, deduplication, and checking against the webpages already gathered (a webpage should not be searched twice), and some Brockport specific filters. Once that is all complete, we feed those links back in to the original function - hence the recursion. Also, importantly, there is a 1-3 second delay between calls to each webpage. This is done to respect the usability of the website for others. See the image below for a visual:

### **Should probably update this to make it better**
![Data Generation Process](https://i.ibb.co/KjnySP1/cycle-white.png)

Since this happens recursively after each loop there should be more links (until we've depleted them all). For increasigly high depths then we should approach the true number of webpages on the Brockport website.

On my first iteration of this, the number of webpages per 1 depth is as follows (originating from [SUNY Brockport Homepage](https://www2.brockport.edu/)):

| Depth | Number of Additional Links | Time Elapsed in Pass |
|-------|----------------------------|----------------------|
| 1     | 1                          | 2 seconds            |
| 2     | ~30                        | ~50 seconds          |
| 3     | ~500                       | ~8 minutes           |
| 4     | ~2500                      | ~2 hours             |
| 5     | ~2500                      | ~2 hours             |

In this case, I only went up to a depth of 5 since that covered the majority of the website, and all of what I was interested in. In total, there were 5413 webpages scraped. Higher depths would have given more data, but was overwhelmingly professor websites and other things I'm not interested in for this project.

---
## Data Cleaning

After scraping the website initially it quickly became clear that there was work to be done cleaning it. For one, the raw HTML files themselves do not help with either of finetuning or retreival augmented generation (RAG). But also there was quite a bit of duplication in the dataset from single webpages having multiple URLs pointing to it.

The full cleaning process and details are in [full_dataset_cleaning.py](./full_dataset_cleaning.py), but at a high level, these are the steps taken to clean the dataset (in order):

- Remove failed webpage retreivals (error 404s, etc.)
- Use [trafilatura](https://trafilatura.readthedocs.io/en/latest/) to parse the HTMLs for data
- Deduplicate dataset by data portion
- Parse all data in www2.brockport.edu/live/.../... and filter to only school policies
- Filter out webpages with little data (<275 characters)
- Standardize URLs by changing _ to - and getting rid of suffix's
- Deduplicate URLs after standardization
- Remove specific webpages I decided are bad
- Save cleaned data to CSV

This cleaning process about half of the webpages in the dataset, reducing its size from 5413 to 2577 webpages. 

---
## Categorize Dataset by URL

After cleaning the dataset, I wanted to categorize the webpages by their URL. This is because I wanted to be able to filter the dataset by URL, and also because I wanted to be able to generate data from specific topics. For example, I wanted to be able to generate data from only the financial aid webpages, or only the admissions webpages. This is done in [categorize_urls.py](./categorize_urls.py), and afterwards is only used in [categorized_engine_setup.py](../chatgpt_pe/categorized_engine_setup.py) to improve the search engine.

---
## Convert to QA Dataset Using GPT3.5 (ChatGPT)

After all the initial data collection, we will have the raw HTML contents of a grand majority of webpages on the Brockport website. From here we need to convert this into question/answer format to eventually fine tune some undetermined LLM. Additionally, for the scratch model, this data will be used to try and train it to begin with (though with more, probably). 

As of June 2023, one common way to change this raw data into more usable data is with GPT3.5 (chatGPT). By leveraging the OpenAI API this process can be made possible at scale. In this implementation, I have about 5600 webpages from the SUNY Brockport website. This makes using a tool like GPT3.5 both beneficial and necessary. 

Processes like this have been done similarly with other models, such as [Stanfords Alpaca](https://github.com/tatsu-lab/stanford_alpaca). It and similar models have taken GPT3.x outputs to fine tune models like [Metas LLaMA](https://github.com/facebookresearch/llama), which demonstrates this idea is proven to work.

The key idea of what I've done in this step is to run up to 64 calls to the OpenAI API concurrently. In the prompt, I've requested that GPT3.5 outputs the data in json format, with up to 25 question/answer combinations according to the "clean" HTML (using BeautifulSoup). In reality, this ends up being "json-like" moreso than actual json. So to parse the responses given I've implemented regex search for "instruction" (question) and "output" (answer). This is passed in to a python list of dictionaries which is then dumped to a proper json (aptly named gpt_output_json.json). I've chosen to model the format of the json file after the aforementioned Alpaca model. This should simplify the training process in the long run.

---
## Results

At depth 5, and originating from the [SUNY Brockport Homepage](https://www2.brockport.edu/), there are 5413 webpages I've scraped. At a depth of 6, there would be an additional 1187 webpages as well. In total there were 113,862 question/answer combinations generated. Here is an example of the json output for 3 of the questions:

```
[
    ...,{
        "instruction": "What is the cost of a double room for the Fall 2023 semester?",
        "output": "The cost of a double room for the Fall 2023 semester is $4,765."
    },
    {
        "instruction": "How can I visit SUNY Brockport?",
        "output": "You can visit SUNY Brockport through a map or take a virtual tour on their website."
    },
    {
        "instruction": "Where can I find the faculty and staff directory?",
        "output": "The faculty and staff directory can be found on the college website."
    },...
]
```

According to the OpenAI website, this dataset accounts for ~13,000,000 tokens between input/output. As of June 18, 2023, their API pricing for gpt-3.5-turbo-0613 is $0.0015/1k for input tokens and $0.002/1k output tokens. The total cost of generating this dataset was $24.18.

---
## Concluding thoughts on first implementation

Whether or not this is the best way to attack this problem needs more testing. It is very possible - frankly, very likely - that using this method will make the data generated highly specific since there are an abundance of webpages that not many people use. Professor webpages for instance, many of which are almost completely blank, do account for a somewhat significant portion of the data. More exploration could be done here, and for the future we might consider filtering the dataset down to only questions generated from specific topics. Maybe admissions and financial aid questions, for example. This is all pendent on the models generated from this dataset though, and needs considerabely more testing to know further.

Considering this possibility however, I've made sure to save checkpoints throughout this code which specifically include URLs and their data. The first is the URL/HTML dictionary, and the second is a URL/GPT_question dictionary (neither available on GitHub, they are fairly large). If needed then, it will be possible to filter URLs to more important topics without regenerating the data from scratch.

#### September 2023 Update

It appears that through the dilligent cleaning steps taken this dataset seems to be pretty good.