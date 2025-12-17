import json
import os

input_file = 'News_Category_Dataset_v3.json'
output_file = 'dataset.json'

# Category mapping dictionary
category_mapping = {
    # Arts & Entertainment
    "ARTS": "Arts & Entertainment",
    "ARTS & CULTURE": "Arts & Entertainment",
    "COMEDY": "Arts & Entertainment",
    "CULTURE & ARTS": "Arts & Entertainment",
    "ENTERTAINMENT": "Arts & Entertainment",
    "STYLE": "Arts & Entertainment",
    "STYLE & BEAUTY": "Arts & Entertainment",
    "MEDIA": "Arts & Entertainment",
    "WEIRD NEWS": "Arts & Entertainment",

    # Society & Identity
    "BLACK VOICES": "Society & Identity",
    "LATINO VOICES": "Society & Identity",
    "QUEER VOICES": "Society & Identity",
    "WOMEN": "Society & Identity",
    "RELIGION": "Society & Identity",

    # Politics & World Affairs
    "POLITICS": "Politics & World Affairs",
    "WORLD NEWS": "Politics & World Affairs",
    "THE WORLDPOST": "Politics & World Affairs",
    "WORLDPOST": "Politics & World Affairs",
    "U.S. NEWS": "Politics & World Affairs",
    "CRIME": "Politics & World Affairs",
    "IMPACT": "Politics & World Affairs",

    # Business & Economy
    "BUSINESS": "Business & Economy",
    "MONEY": "Business & Economy",

    # Science, Tech & Environment
    "SCIENCE": "Science, Tech & Environment",
    "TECH": "Science, Tech & Environment",
    "ENVIRONMENT": "Science, Tech & Environment",
    "GREEN": "Science, Tech & Environment",

    # Health & Wellness
    "WELLNESS": "Health & Wellness",
    "HEALTHY LIVING": "Health & Wellness",

    # Lifestyle & Leisure
    "FOOD & DRINK": "Lifestyle & Leisure",
    "TRAVEL": "Lifestyle & Leisure",
    "TASTE": "Lifestyle & Leisure",
    "HOME & LIVING": "Lifestyle & Leisure",
    "WEDDINGS": "Lifestyle & Leisure",
    "DIVORCE": "Lifestyle & Leisure",
    "SPORTS": "Lifestyle & Leisure",
    "GOOD NEWS": "Lifestyle & Leisure",
    "FIFTY": "Lifestyle & Leisure",

    # Education & Youth
    "EDUCATION": "Education & Youth",
    "COLLEGE": "Education & Youth",
    "PARENTS": "Education & Youth",
    "PARENTING": "Education & Youth"
}

categories = set()
unmapped_categories = set()

try:
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        # Write the start of the JSON array
        f_out.write('[\n')
        
        first_line = True
        for line in f_in:
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                
                # Remove fields
                if 'link' in data:
                    del data['link']
                if 'date' in data:
                    del data['date']
                
                # Map category
                if 'category' in data:
                    original_category = data['category']
                    if original_category in category_mapping:
                        data['category'] = category_mapping[original_category]
                    else:
                        unmapped_categories.add(original_category)
                        # Keep original if not mapped, or could map to "OTHERS"
                        # For now, keeping original to see what's missed
                    
                    categories.add(data['category'])
                
                # Write to output file
                if not first_line:
                    f_out.write(',\n')
                
                json.dump(data, f_out, ensure_ascii=False)
                first_line = False
                
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line[:50]}...")
        
        # Write the end of the JSON array
        f_out.write('\n]')

    print("数据清洗与类别合并完成。")
    print("\n新的分类列表：")
    for category in sorted(list(categories)):
        print(category)
        
    if unmapped_categories:
        print("\n未映射的原始分类（请检查是否需要添加到映射表）：")
        for category in sorted(list(unmapped_categories)):
            print(category)

except FileNotFoundError:
    print(f"错误：找不到文件 {input_file}")
except Exception as e:
    print(f"发生错误：{e}")
