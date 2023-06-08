# Run `copy(JSON.stringify(Array.from(document.querySelectorAll("li")).map((e)=>e.textContent.trim().toLowerCase().replace('.','')).slice(18, 686).filter((e)=>!e.includes("địa điểm")).filter((e)=>!e.includes("đi đâu")).filter((e)=>!e.includes("?")), null, 4))` on http://vatc.vn/vi/tin-tuc/du-lich-viet-nam/1000-diem-du-lich-hap-dan-o-63-tinh-thanh-cua-viet-nam/ to get the list of places and specialties and save it to `places_and_specialties.json`


import json
from icrawler.builtin import GoogleImageCrawler

with open("places_and_specialties.json", encoding="utf-8") as f:
    items = json.load(f)

for i in range(len(items)):
    google_crawler = GoogleImageCrawler(storage={"root_dir": f"images/{i}"})
    google_crawler.crawl(keyword=items[i], max_num=2)
