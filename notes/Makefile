NAME=plan
TEMP=template.html

all:
	cat $(TEMP) > $(NAME).html
	markdown $(NAME).md >> $(NAME).html
	echo "</body></html>" >> $(NAME).html
