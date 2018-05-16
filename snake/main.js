//Delcare Global Variables
var s;
var size  = 20;
var food;
var count=0;
function setup() {
  count=0;
  createCanvas(550, 550);
  background(0);
  s = new Snake();
  food_location();
  frameRate(15);

}
function food_location()
{
var rows = 550 ;
var columns = 550 ;
food = createVector(random(rows),random(columns));
}
function draw() {
  background(0);
  s.move();
  s.show();
  s.die();
  if (s.eat_food() == true)
  {
  count++;
  fill(0);
  console.log(count,s.tail);
  rect(food.x,food.y,20,20);
  food_location();
  }
  fill (255,100,100);
  rect(food.x,food.y,20,20);
}

function Snake() {
  this.tail = [];
  this.x =0;
  this.y =0;
  this.xspeed = 1;
  this.yspeed = 0;

  this.move = function(){
    if (count === this.tail.length) {
      for (var i = 0; i < this.tail.length-1; i++) {
        this.tail[i] = this.tail[i+1];
    }

    }
    this.tail[count-1] = createVector(this.x, this.y);


    this.x = this.x + this.xspeed*20;
    this.y = this.y + this.yspeed*20 ;
  }

  this.eat_food = function()
  {
    var distance = dist(this.x,this.y,food.x,food.y);
    if(distance < 15)
    {
    console.log("FOOD");
    return true;

    }
    return false;
  }

  this.show = function(){
    fill(255);
    for(var i=0;i<count;i++)
    {
     rect(this.tail[i].x, this.tail[i].y, 20,20);
    }
    rect(this.x, this.y, 20,20);
  }
  this.chng_dir = function(x,y)
    {
     this.xspeed = x;
     this.yspeed = y;
    }
  this.die = function()
  {
   if(this.x > 600 || this.y >600 || this.x <0 || this.y <0)
   {
    setup();
//    this.yspeed =0;
//    this.xspeed =0;
//
//    this.x =0;
//    this.y=0;
//    food_location();
//    this.show();
   }
  }
}
function keyPressed()
{
    if (keyCode === UP_ARROW)
    {
        s.chng_dir(0,-1);
    }
    else if (keyCode === DOWN_ARROW)
    {
        s.chng_dir(0,1);
    }
    if (keyCode === RIGHT_ARROW)
    {
        s.chng_dir(1,0);
    }
    if (keyCode === LEFT_ARROW)
    {
        s.chng_dir(-1,0);
    }
}
