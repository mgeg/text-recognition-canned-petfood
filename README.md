# Text Recognition From Canned Cat/Dog Food

Sparked by our daily life, many people are pets lovers. As a responsible pet owner, when picking the canned food for pets, we care about the nutrition/ingredients. However, the thing is that the can is very small, and the ingredients list on the can is tiny. It’s hard and time consuming for us to recognize the important main ingredients or the artificially added materials, especially for those allergic pet owners.

The project aims to generate the keyword summary from the given photo of the pet food label. We first use text recognition algorithms to grab words from images. Then use NLP and string patterns to obtain the label summary.

# Data
As there are no prepared data set for pet food label images, we grab the data from online web pages. As the product intro page has ingredient lists, we manually use those images as our feed data. We collected 77 images of the canned pet food labels in total, stored at [/data](https://github.com/mgeg/text-recognition-canned-petfood/tree/main/data). 

# Process
Here is the sample pipeline of our works.

<table><tr><td>
<img src="images/process.png" width="550">
</td></tr></table>

### Step 1. Text Recognition



### Step 2. Keyword Extraction



### Step 3. String Pattern Matching


# Conclusion
