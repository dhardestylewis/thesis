import glob
files = glob.glob('Analysis/Scripts/Warehouse_Builder/*.py') + glob.glob('scripts/*.py')
remove_str = "'bisg_white_200ft', 'bisg_black_200ft', 'bisg_asian_200ft', 'bisg_hispanic_200ft', 'bisg_white_nbr', 'bisg_black_nbr', 'bisg_asian_nbr', 'bisg_hispanic_nbr', "

for f in files:
    try:
        with open(f, 'r', encoding='utf-16') as file:
            content = file.read()
        encoding = 'utf-16'
    except UnicodeError:
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
        encoding = 'utf-8'
    
    if remove_str in content:
        content = content.replace(remove_str, '')
        with open(f, 'w', encoding=encoding) as file:
            file.write(content)
        print(f"Reverted {f}")
