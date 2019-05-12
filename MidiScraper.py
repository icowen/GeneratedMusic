from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from time import sleep

driver = webdriver.Chrome("C:/Intel/chromedriver.exe")
driver.get("http://www4.osk.3web.ne.jp/~kasumitu/eng.htm")
elems = driver.find_elements_by_tag_name("a")
file_type = ".mid"
for option in elems:
    if option.get_attribute("href")[-4:] == file_type:
        print(option.get_attribute("href"))
        option.click()
        sleep(1)

