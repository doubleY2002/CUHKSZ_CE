import random
import turtle
import time

table=turtle.Screen()
table.setup(600,680)
table.tracer(0)

line=turtle.Turtle()
word=turtle.Turtle()
s=turtle.Turtle()
m=turtle.Turtle()
contact=0
flag=0
click=0
start_time=0
absx=500
absy=500

food_list=[]
food=[]

pause_list=[(0,0)]
direction=540
direction_list=[]

stamp_list=[]
snake_pos=[]
length=5

def up():
    global direction,direction_list
    direction=90
    direction_list.append(90)
def down():
    global direction,direction_list
    direction=270
    direction_list.append(270)
def right():
    global direction,direction_list
    direction=0
    direction_list.append(0)
def left():
    global direction,direction_list
    direction=180
    direction_list.append(180)

def pause():
    global pause_list,direction
    if (s.xcor,s.ycor)!=pause_list[-1]:
        direction=360
        pause_list.append((s.xcor(),s.ycor()))
    else :
        direction=direction_list[-1]

def longer():
    global stamp_list,snake_pos
    s.color("blue","black")
    s.setheading(direction)
    s.forward(20)
    w=s.stamp()
    s.color("red")
    stamp_list.append(w)
    snake_pos.append((s.xcor(),s.ycor()))
    table.update()

def getpos():
    x=random.uniform(-230,230)
    y=random.uniform(-230,230)
    return (x,y)

def set_table():
    table.title("Snake by doubleY")
    line.hideturtle()
    line.pensize(5)
    line.up()
    line.goto(-250,330)
    line.down()
    line.forward(500)
    line.right(90)
    line.forward(580)
    line.right(90)
    line.forward(500)
    line.right(90)
    line.forward(580)
    line.goto(-250,250)
    line.right(90)
    line.forward(500)
    word.hideturtle()
    word.up()
    word.goto(-210,280)
    word.write("Contact:"+str(contact)+"  Time: 0  Motion: Paused",font=("Arial",14,"normal"))

    s.hideturtle()
    s.up()
    s.goto(-200,210)
    s.color("black")
    s.write("Welcome to doubleY's version of snake...",font=("Arial",12,"normal"))
    s.goto(-200,170)
    s.write("You are going to use 4 arrow keys to move the snake",font=("Arial",12,"normal"))
    s.goto(-200,150)
    s.write("around the screen, trying to consume all the food items",font=("Arial",12,"normal"))
    s.goto(-200,130)
    s.write("before the monster catch you...",font=("Arial",12,"normal"))
    s.goto(-200,90)
    s.write("Click anywhere to start the game, have fun!",font=("Arial",12,"normal"))
    s.goto(0,0)
    s.color("red")
    s.shape("square")
    s.showturtle()

    m.up()
    x=random.uniform(1,8)*20+50
    y=random.uniform(1,8)*20+50
    m.goto(x,-y)
    m.shape("square")
    m.color("purple")
    m.showturtle()
    table.update()

def create_food():
    for i in range(9):
        pos=getpos()
        w=turtle.Turtle()
        w.hideturtle()
        w.up()
        w.goto(pos)
        w.write(i+1)
        food_list.append(w)
        food.append(i)

def startgame(x,y):
    global stamp_list,snake_pos,direction,start_time,click
    if click==0:
        direction=360
        s.clear()
        click=1
        start_time=time.time()
        s.color("blue","black")
        stamp_list.append(s.stamp())
        snake_pos.append((s.xcor(),s.ycor()))
        s.color("red")
        create_food()
        table.update()

def eating():
    global food_list,contact
    ans=0
    x=s.xcor()
    y=s.ycor()
    for i in food:
        fx=food_list[i].xcor()
        fy=food_list[i].ycor()
        if abs(fx-x)<=10 and abs(fy-y)<=10:
            food_list[i].clear()
            food_list[i].hideturtle()
            food.remove(i)
            ans+=i+1
    return ans

def snake_moveonce():
    global length,snake_pos
    longer()
    snake_len=len(stamp_list)-1
    length+=eating()
    if length>snake_len:
        table.ontimer(snake_move,400)
    else:
        s.clearstamp(stamp_list[0])
        table.update()
        stamp_list.pop(0)
        snake_pos.pop(0)
        table.ontimer(snake_move,200)

def check_in():#in the table
    if int(s.xcor()) in range(-230,230) and int(s.ycor()) in range(-230,230):
        return True
    else:
        return False

def check_fail():#the monster catch the snake
    if s.distance(m)<=20: return True
    return False

def check_win():#eat all and win
    if len(food)==0: return True
    return False

def contacting():
    for pos in snake_pos:
        if abs(pos[0]-m.xcor())<20 and abs(pos[1]-m.ycor())<20:
            return 1
    return 0

def snake_move():
    if flag==1:return
    if direction != 360 and direction != 540 and check_in():
        snake_moveonce()
    #if the snake is out of the range, only in the following cases can it turns back to the screen
    elif int(s.xcor()) not in range(-230,230) and int(s.ycor()) not in range(-230,230):
        if s.xcor()>230 and s.ycor()<-240:
            if direction==0 or direction==270:
                table.ontimer(snake_move,200)
            else:
                snake_moveonce()
        if s.xcor()<-230 and s.ycor()<-230:
            if direction==180 or direction==270:
                table.ontimer(snake_move,200)
            else:
                snake_moveonce()
        if s.xcor()<-230 and s.ycor()>230:
            if direction==90 or direction==180:
                table.ontimer(snake_move,200)
            else:
                snake_moveonce()
        if s.xcor()>230 and s.ycor()>230:
            if direction==0 or direction==90:
                table.ontimer(snake_move,200)
            else:
                snake_moveonce()
    elif direction!=180 and s.xcor()<-230:
        snake_moveonce()
    elif direction!=270 and s.ycor()<-230:
        snake_moveonce()
    elif direction!=0 and s.xcor()>230:
        snake_moveonce()
    elif direction!=90 and s.ycor()>230:
        snake_moveonce()
    #if the game is pause, just need to update the situation right now
    else:
        table.ontimer(snake_move,200)

def monster_move():
    if flag==1: return
    if direction!=540:
        global absx,absy
        absx=s.xcor()-m.xcor()
        absy=s.ycor()-m.ycor()
        if abs(absx)>=abs(absy) and absx>0:
            m.setheading(0)
            m.forward(20)
        if abs(absx)>=abs(absy) and absx<0:
            m.setheading(180)
            m.forward(20)
        if abs(absy)>abs(absx) and absy>0:
            m.setheading(90)
            m.forward(20)
        if abs(absy)>abs(absx) and absy<0:
            m.setheading(270)
            m.forward(20)
    table.update()
    table.ontimer(monster_move,320)

def work():
    table.onkey(right,"Right")
    table.onkey(up,"Up")
    table.onkey(left,"Left")
    table.onkey(down,"Down")
    table.onkey(pause,"space")
    table.onclick(startgame)
    table.listen()

def motion():
    if direction==0: return "Right"
    if direction==90: return "Up"
    if direction==180: return "Left"
    if direction==270: return "Down"
    return "Paused"

def change():
    global flag,contact
    if flag==1:return
    if direction!=540:
        tim=int(time.time()-start_time)
        contact+=contacting()
        if len(food)==0:tim=0
        word.hideturtle()
        word.clear()
        word.up()
        word.goto(-210,280)
        word.write("Contact:"+str(contact)+"  Time:"+str(tim)+"  Motion:"+motion(),font=("Arial",14,"normal"))
        table.update()
        if check_fail():
            word.hideturtle()
            word.goto(0,0)
            word.color("purple")
            word.write("Game over!",font=("Arial",12,"normal"))
            table.update()
            flag=1
            return
        elif tim>3 and check_win():
            word.hideturtle()
            word.goto(0,0)
            word.color("red")
            word.write("Winner!",font=("Arial",12,"normal"))
            table.update()
            flag=1
            return
    table.ontimer(change,200)

def main():
    set_table()
    snake_move()
    monster_move()
    change()
    work()
    turtle.done()

main()