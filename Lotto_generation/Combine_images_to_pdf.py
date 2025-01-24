import glob
import img2pdf
import pandas as pd

parent_folder = 'V4.3_Lotto_run/'
file_name = 'EPN_sources'

Process_Lotto = True
Process_Colour_Mag = False

Objects_list_csv = 'EPN_sources_from_lotto.csv'


if(Objects_list_csv is not None):
    Objects_list = [str(current) for current in pd.read_csv('{}{}'.format(parent_folder, Objects_list_csv))['Index'].values]
else:
    Objects_list = None

print(Objects_list)    

if(Process_Lotto):
    file_names = glob.glob("{}Images with C1/*.png".format(parent_folder))

    file_names.sort()
    
    print(file_names)
    
    if(Objects_list is not None):
        
        file_names_new = []
        
        for current in file_names:
            print(current)
            print(current[len(parent_folder) + 15 : -6])
            
            tmp = int(current[len(parent_folder) + 15 : -6])
            
            print(tmp, str(tmp))
            
            if(str(tmp) in Objects_list):
                file_names_new.append(current)
        
        file_names = file_names_new
        
        print(len(file_names))
    
    with open("{}{}.pdf".format(parent_folder, file_name), "wb") as f:
        f.write(img2pdf.convert(file_names))

if(Process_Colour_Mag):
    file_names = glob.glob("{}Colour_Mag_Images/*.png".format(parent_folder))

    file_names.sort()
    
    if(Objects_list is not None):
        
        file_names_new = []
        
        for current in file_names:
            if(current[len(parent_folder) + 11 : -4] in Objects_list):
                file_names_new.append(current)
        
        file_names = file_names_new

    with open("{}{}_Colour_Mag.pdf".format(parent_folder, file_name), "wb") as f:
        f.write(img2pdf.convert(file_names))

