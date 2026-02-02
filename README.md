# Associative-Classification-Project
Corporate Credit Evaluation Project Using Big Data Analysis

[한국어 🇰🇷](README.ko.md)

## The Purpose of This Project
  + Utilize corporate review data for corporate credit evaluation through big data analysis
  + Perform reliability validation for using corporate review data through data modeling

## Members of This Project
  + Solo project

## The Process of This Project
  + Scrape corporate review data by accessing the Jobplanet website
  + Perform data preprocessing and split the data into structured and unstructured data
  + Structured data consists of review ratings (overall score, welfare score, culture score, diversity score, etc.)
  + Unstructured data consists of text such as pros, cons, and suggestions to management
  + Perform data classification on structured data
  + Perform NLP on unstructured data
  + If the data model accuracy exceeds 90%, corporate review data is used for credit evaluation

## Structured Data Collection and Analysis
  + Data collection
    + Download a driver for accessing Chrome
    + Load the downloaded driver
      ```
      driver = webdriver.Chrome("/Users/eungchan/desktop/apply/chromedriver")
      ```
    + Access the Jobplanet website using the driver
      ```
      driver.get("https://www.jobplanet.co.kr/users/sign_in?_nav=gb")
      ```
    + Log in using a pre-registered ID and password, then navigate to the review page
      ```
      login_id = driver.find_element_by_css_selector("input#user_email")
      login_id.send_keys(usr)
      login_pwd = driver.find_element_by_css_selector("input#user_password")
      login_pwd.send_keys(pwd)
      login_id.send_keys(Keys.RETURN)
      
      search_query = driver.find_element_by_css_selector("input#search_bar_search_query")
      search_query.send_keys(query)
      search_query.send_keys(Keys.RETURN)
      ```
    + Crawl and preprocess data
  + Implement a decision tree model using 90% of the entire dataset as training data
  + The model can predict the following:
  + If promotion opportunity = AA, welfare = BB, work-life balance = A, and corporate culture = AA, then the final grade is AA
  + Use the remaining 10% of the data as test data and store the final grade separately
  + If the review scores of test data without a final grade match the newly generated review scores by more than 90%, the reliability of the corporate review data is verified

## Data Modeling
  + Build a model using a Decision Tree
  + Build the tree using information gain
    ```
    def get_InfoGain(data, sub_root, target_name):
        # Calculate total entropy
        total_entropy = get_entropy(data[target_name])

        # Calculate entropy for each subtree
        # Extract unique attribute values and their counts
        sub_trees, counts = np.unique(data[sub_root], return_counts=True)
        # Calculate probabilities
        p = get_prob(counts)

        # Calculate weighted entropy for the sub_root
        sub_entropy = 0
        for i in range(len(sub_trees)):
            sub_entropy += p[i] * get_entropy(
                data.where(data[sub_root] == sub_trees[i]).dropna()[target_name])

        Info_Gain = total_entropy - sub_entropy
        return Info_Gain
    ```
  + The loss function used to calculate information gain is entropy
    ```
    def get_entropy(an_attribute):
        # Return unique values and their counts for the given attribute
        unique_elements, counts = np.unique(an_attribute, return_counts=True)
        # Calculate probabilities
        p = get_prob(counts)

        # Calculate entropy
        entropy = 0
        for i in range(len(unique_elements)):
            entropy += -(p[i] * np.log2(p[i]))
        return entropy
    ```

## Model Evaluation Results

<img width="793" alt="Model Evaluation Result" src="https://user-images.githubusercontent.com/83147205/147481887-c9ba94a7-e37e-4245-835f-bd79fee7b4c0.png">
