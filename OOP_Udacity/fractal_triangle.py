# import turtle

# def draw_triangle():

# 	window=turtle.Screen()

# 	window.bgcolor("orange")

# 	araz=turtle.Turtle()
# 	araz.shape("turtle")
# 	araz.speed("fastest")
# 	araz.setpos(0,0)

# 	for i in range(2):

# 		araz.color("black","blue")
# 		araz.begin_fill()
# 		araz.forward(50)
# 		araz.left(120)
# 		araz.forward(50)
# 		araz.left(120)
# 		araz.forward(50)
# 		araz.left(120)
# 		araz.end_fill()
# 		#araz.fillcolor("blue")
# 		#araz.forward(100)
# 		#araz.left(5)
# 		araz.left(60)
# 		araz.forward(50)
# 		#araz.reset()
# 	window.exitonclick()	
		



# #draw_sqaure()

# #for i in range(36):

# draw_triangle()


import turtle
#Haha TTpro
def draw_triangle(the_turtle,length,ori,recursion):
    recursion=recursion+1
    meow= the_turtle

    for i in range(0,3):
        if(recursion<4):
        #if (0):
            meow.forward(length/2)
            meow.left(120)
            draw_triangle(meow,length/2,0,recursion)
            meow.right(120)
            meow.forward(length/2)
        else:
            meow.forward(length)
        if (ori==1):
            meow.left(120)
        else:
            meow.right(120)

meow = turtle.Turtle() # init
meow.speed(10) # speed = 1 to slow turtle down
meow.color("yellow","green") # set color5
meow.shape("turtle") # set shape to a turtle
background = turtle.Screen()  # create background
background.bgcolor("red")     # set background color to red


draw_triangle(meow,200,1,0)

#meow.forward(100)
#meow.left(120)
#draw_triangle(meow,100,0,0)
#meow.right(120)

background.exitonclick()      # click anywhere to close background