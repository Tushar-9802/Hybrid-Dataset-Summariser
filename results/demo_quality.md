# Generation-quality check — base vs phase3 adapter

Generated on 4 test-split videos. Decoding: greedy, max_new_tokens=160, no_repeat_ngram_size=3.

## Sample 1 — `p_VxqEBiNiA`

**Source transcript (truncated):**
> In today's video we are going to discuss about KV charts. Actually what is mean by KV chart? KV chart is nothing but a Boolean algebraic expressions are used to determine which cases are interesting and which combination of predicate value should be used to reach which node. Sizes of kv chart is from 2 variable to 5 and this This kv chart reduces boolean algebraic manipulations to graphical trivia. Firstly we discuss about one variable kv chart here 0 a 0 1 this function never true. This function is true when a is 1 this function is true when a is 1 this function is true when a is false here t...

**Gold reference:**
> The video explains KV charts, which are graphical tools used to simplify Boolean algebra expressions. It starts with one-variable charts and progresses to two, three, and four-variable charts, detailing how to represent combinations of variable states. The presenter demonstrates how to pair values and derive equations from the charts, emphasizing practical examples and the conversion of decimal numbers into binary representations for KV chart mapping.

**Base Mistral-7B** (9.7s · PVR=0.00  Nom=0.000  TTR=1.000  ASL=53.0  len=53w · R1=0.000  R2=0.000  RL=0.000):

```
10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62
```

**+ Phase-3 adapter** (15.7s · PVR=0.00  Nom=0.000  TTR=1.000  ASL=53.0  len=53w · R1=0.000  R2=0.000  RL=0.000):

```
10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62
```

## Sample 2 — `8xUher8-5_Q`

**Source transcript (truncated):**
> AI is changing extremely fast in 2025, and so is the way that you should be learning it. So in this video, I'm going to break down exactly how I would learn AI and ML if I was starting completely from scratch with all of the knowledge that I have today. Let's get into it. Now, the first thing or step zero on my list would be to make sure that I was thinking like an engineer. Now, look, there's a long list of topics that I'm going to share with you here, all things that are important to learn, but none of them matter if you don't build that deep critical thinking skill. The things that separate...

**Gold reference:**
> The video outlines a practical approach to learning AI and machine learning from scratch. It emphasizes the importance of thinking like an engineer and developing critical problem-solving skills before diving into specific technologies. The speaker recommends starting with Python for automation and data manipulation, using libraries like NumPy, Pandas, and Matplotlib, and getting comfortable with APIs. Next, they suggest becoming data literate by learning SQL and advanced Pandas operations. The video then encourages immediate hands-on experience with AI models using tools like OpenAI API and Langchain, followed by mastering core ML algorithms such as regression and classification with scikit-learn. Finally, it stresses the importance of building numerous AI applications to solidify skills and apply knowledge in real-world scenarios.

**Base Mistral-7B** (9.4s · PVR=0.00  Nom=0.000  TTR=0.713  ASL=21.5  len=129w · R1=0.216  R2=0.023  RL=0.124):

```
So again, this isn't about memorizing a bunch of concepts. This is about being able to understand what's happening under the hood and being able apply those concepts to solving problems. So once I had done that, I would then move onto step three, which is to learn the math behind AI and ML, and this is where things get a little bit trickier. Look, I'll be honest with you. There's a ton of different ways to go about this, and I'm not even sure that I would recommend this to everyone. But if you've already gone through steps one and two, and you're looking to get really good at AI and ML and you've got time to invest, then I would highly consider taking a deep dive into
```

**+ Phase-3 adapter** (16.0s · PVR=0.00  Nom=0.125  TTR=0.646  ASL=32.2  len=130w · R1=0.254  R2=0.023  RL=0.108):

```
Now, once I've gotten comfortable with data, I would then shift my attention attention attention attention to attention attention attention To be honest, I've been struggling with this one myself recently because I feel like I've lost my touch with the community. So what I've decided to do is to join a local meetup group and attend as many events as possible to reconnect with other developers and AI enthusiasts. I've found that this has helped me tremendously in staying up-to-date with the latest trends and techniques in in in in AI and ML learning learning learning learning. Another thing that I've started doing is volunteering for coding clubs at my local university, which has given me the opportunity to mentor young developers and share my experiences with them. Finally
```

## Sample 3 — `7h1s2SojIRw`

**Source transcript (truncated):**
> Hi, the topic is quicksort. First of all let us understand what is the idea behind quicksort on what basis it works what is the base idea about it. So let us take an example so the idea is if suppose there are group of students in a class and a teacher asked them to arrange themselves in the increasing order of their height there are two options a teacher can show their places like you go there and stand at the back you come here and stand in the middle or front whatever it is so the teacher can show places to the students this is one option second option is teacher can ask the student to arra...

**Gold reference:**
> The video explains the quicksort algorithm, starting with the concept of students arranging themselves by height as an analogy for sorting. It details the partitioning process, where a pivot is selected, and elements are rearranged so that those smaller than the pivot are on one side and those larger are on the other. The presenter walks through the algorithm step-by-step, demonstrating how to implement it recursively, and highlights the importance of the partitioning method in achieving the sort.

**Base Mistral-7B** (9.2s · PVR=0.00  Nom=0.076  TTR=0.435  ASL=10.8  len=131w · R1=0.295  R2=0.048  RL=0.162):

```
. Now I have to partition the list around this pivot. So I have to move all the elements less than pivot to the left of pivot and all the element greater than pivot on the right of pivot. Let us see how I will do that. So I start with the first element 6. It is less than pivotal so I will put it on the left of the pivot. Then I will compare next element 3. It is also less than pivalot so I will keep it on the same side. Next element is 5. It is again less than pivalent so I will again keep it on this side. Next is 4. Again less than pivenal so I will continue putting it on the side. Next one is
```

**+ Phase-3 adapter** (15.6s · PVR=0.33  Nom=0.071  TTR=0.635  ASL=21.0  len=63w · R1=0.306  R2=0.070  RL=0.167):

```
quick sort is based on the concept that an element can easily find its sorted position in in in in a sorted list. So the idea is that an element should be able to find its sorted positioned without much effort. So the teacher' the video explains quicksort, detailinginginging the algorithm algorithm algorithm algorithmalgorithm algorithm algorithm algorithmgorithm algorithm algorithm algorithm algebraicicicicpicicicic picicicic icicicicicipiciciciciicicicciiciciciceicicicieiciciccieicicicieceiciciciciaicicicicioicicicicoiciciccioicicicicióicicicicionicicicicionesiciciciciosicicicicosiciciciciesiciciciciousicicic
```

## Sample 4 — `P5ihIzoZivc`

**Source transcript (truncated):**
> Hopefully you have had a chance gotten some emails from me about the resources that we're going to use today and these are the packages we're going to use today. Having said that I always like to start with the Duke University land acknowledgement so I would take just a moment if you will give me that privilege to honor the land in Durham North Carolina Duke University sits on the ancestral land of the Shikori, the Eno, and the Catawba people. This institution of higher education is built on the land stolen from those people. These tribes were here before the colonizers arrived. Additionally, ...

**Gold reference:**
> The video focuses on using R for text analysis, particularly through the Tidyverse and the TidyText package. It starts with data cleaning techniques, emphasizing the importance of preparing text data for analysis. Key concepts include tokenization, where text is split into meaningful units, and the creation of a document-term matrix. The presenter demonstrates sentiment analysis using the Bing sentiment dictionary, illustrating how to calculate sentiment scores and visualize results with ggplot. Additionally, the video touches on the TD-IDF approach for word frequency analysis and discusses various sentiment dictionaries like NRC and AFIN. Throughout, practical coding examples are provided, using Jane Austen's works as the text corpus.

**Base Mistral-7B** (4.7s · PVR=0.17  Nom=0.000  TTR=0.737  ASL=9.5  len=57w · R1=0.107  R2=0.000  RL=0.083):

```
I guess early 1800s. And she was a woman who lived in England and her father was a clergyman. And so she didn't have much money, but she was able to live off of her writing. And so I think that's pretty cool. And she's written some great stories. So I encourage you to check them out.
```

**+ Phase-3 adapter** (15.7s · PVR=0.14  Nom=0.041  TTR=0.761  ASL=15.6  len=109w · R1=0.489  R2=0.201  RL=0.262):

```
I forget when exactly, but they were published back back back back in in in in the 18th century. So we're gonna use her works for our text analysis today. And with that let's jump right in.

The video dives into text analysis using R, specifically focusing on the Tidyverse and the Jane Austen collection. It starts with data cleaning techniques, explaining how to remove unwanted characters and standardize text using regular expressions. The instructor emphasizes the importance of token token token tokenizationizationizationization, which involves breaking down text into meaningful units, such as words or characters.

Next, the video covers word counts, introducing the concept of TF----IDF (Term Frequency-----------
```
