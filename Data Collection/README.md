# Data Collection

Data Collection is of key importance for this project. Data must be contextually relevant, and deliver good examples for the models to use.

The primary source of data at our disposal comes directly from the SUNY Brockport website (www2.brockport.edu). It may be possible, if necessary, to request more information from the relevant people, however for the first iteration of data generation I suspect we have enough.

### **First iteration:**

---
## Scraping Brockport Website:
Using data from the website requires us to scrape the site. I've implemented a version of this in [brockport-scraping.ipynb](brockport-scraping.ipynb). In that file, data is collected recursively. From some set of links (or 1, https://www2.brockport.edu/, in my case) all their webpages are retreived. From there I collect all the links on each of those webpages, and apply a set of filters. Most importantly, these filters include: obeying robots.txt, deduplication, and checking against the webpages already gathered (a webpage should not be searched twice), and some Brockport specific filters. Once that is all complete, we feed those links back in to the original function - hence the recursion. Also, importantly, there is a 1-3 second delay between calls to each webpage. This is done to respect the usability of the website for others. See the image below for a visual:

![Data Generation Process](https://i.ibb.co/JQjCLYm/cycle.png)

Since this happens recursively after each loop there should be more links (until we've depleted them all). For increasigly high depths then we should approach the true number of webpages on the Brockport website. 

On my first iteration of this, the number of webpages per 1 depth is as follows (starting from https://www2.brockport.edu/):

| Depth | Number of Additional Links | Time Elapsed in Pass |
|-------|----------------------------|----------------------|
| 1     | 1                          | 2 seconds            |
| 2     | ~30                        | ~50 seconds          |
| 3     | ~500                       | ~8 minutes           |
| 4     | ~2500                      | ~2 hours             |
| 5     | ~2500                      | ~2 hours             |

---
## Convert to QA Dataset Using GPT3.5 (ChatGPT)

In this implementation, after this step, we will have the raw HTML contents of a grand majority of webpages on the Brockport website. From here we need to convert this into question/answer format to eventually fine tune some undetermined LLM. Additionally, for the scratch model, this data will be used to try and train it to begin with (though with more, probably). 

As of June 2023, one common way to change this raw data into more usable data is with GPT3.5 (chatGPT). By leveraging the OpenAI API this process can be made possible at scale. In this implementation, and on this first iteration, I have about 5600 webpages from the SUNY Brockport website. This makes using a tool like GPT3.5 both beneficial and necessary. 

Processes like this have been done similarly with other models, such as [Stanfords Alpaca](https://github.com/tatsu-lab/stanford_alpaca). It and similar models have taken GPT3.x outputs to fine tune models like [Metas LLaMA](https://github.com/facebookresearch/llama), which demonstrates this idea is proven to work.

The key idea of what I've done in this step is to run up to 64 calls to the OpenAI API concurrently. In the prompt, I've requested that GPT3.5 outputs the data in json format, with up to 25 question/answer combinations according to the "clean" HTML (using BeautifulSoup). In reality, this ends up being "json-like" moreso than actual json. So to parse the responses given I've implemented regex search for "instruction" (question) and "output" (answer). This is passed in to a python list of dictionaries which is then dumped to a proper json (aptly named gpt_output_json.json). 

---
## Concluding thoughts on first implementation:

Whether or not this is the best way to attack this problem needs more testing. It is very possible - frankly, very likely - that using this method will make the data generated highly specific since there are an abundance of webpages that not many people use. Think professor webpages, maybe, which do account for a significant portion of the data. More exploration could be done here, and for the future we might consider filtering the dataset down to only questions generated from specific topics. Maybe admissions and financial aid questions, for example. This is all pendent on the models generated from it though, and needs considerabely more testing to know further.

Considering this possibility, I've made sure to save checkpoints throughout this code which specifically include URLs and their data. The first is the URL/HTML dictionary, and the second is a URL/GPT_question dictionary. If needed then, it will be possible to filter URLs to more important topics without regenerating the data from scratch.