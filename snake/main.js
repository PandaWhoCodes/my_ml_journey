//Delcare Global Variables
var s;
var size  = 20;
var food;


function setup() {
  createCanvas(600, 600);
  background(51);
  s = new Snake();
  food_location();
  frameRate(15);


}
function food_location()
{
var rows = 600 /20;
var columns = 600 /20;
food = createVector(random(rows),random(columns));
}
function draw() {
  background(0);
  s.move();
  s.show();
  s.die();
  if (s.eat_food() == true)
  food_location();
  fill (255,100,100);
  rect(food.x,food.y,20,20);
}

function Snake() {
  this.x =0;
  this.y =0;
  this.xspeed = 1;
  this.yspeed = 0;

  this.move = function(){
    this.x = this.x + this.xspeed*20;
    this.y = this.y + this.yspeed*20 ;
  }

  this.eat_food = function()
  {
    var distance = dist(this.x,this.y,food.x,food.y);
    if(distance < 3)
    return true;
    return false;
  }

  this.show = function(){
    fill(255);
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
    this.yspeed =0;
    this.xspeed =0;

    this.x =0;
    this.y=0;
   }
   s.show();
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
