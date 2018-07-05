import os


def rename_files():
	files=os.listdir("prank/")
	print(files)
	print(type(files))

	not_to_have="0123456789"
	dir_="prank/"
	where_=os.getcwd()
	print(where_)

	for i in range(len(files)):

		#os.chdir("file_path")
		x=drop_the_letter(files[i],not_to_have)    #instead can use  translate

		os.rename(str(dir_+str(files[i])),str(dir_+str(x)))



def drop_the_letter(text,not_to_have):

	new_text=""
    
	for j in text:
			if (j in not_to_have):
				pass
			else:
				new_text=new_text=new_text+str(j)		

 
	print(new_text)
	return new_text






rename_files()
