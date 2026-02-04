def bin_search(file_path, bin_number):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if bin_number in line:
                    parts = line.split()
                    
                    for i, part in enumerate(parts):
                        if part == 'S' and i + 1 < len(parts):
                            next_part = parts[i + 1]
                            if '.' in next_part:
                                return next_part
                    
                    return f"Знайдено: {line.strip()}"
        
        return "BIN не знайдено"
    
    except Exception as e:
        return f"Помилка: {str(e)}"

file_path = (r"D:\Робочий стіл\диплом\article 2.0\lc_summary.txt")
bin_to_search = "176917" # вкаазати необхідний BIN в лапках
result = bin_search(file_path, bin_to_search)
print(f"Результат: {result}")
