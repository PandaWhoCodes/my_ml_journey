
var r,g,b;
var ai;
function gen_random_color()
{
 r = random(255);
 g = random(255);
 b = random(255);

}
function mousePressed()
{
gen_random_color();
}
function setup()
{
createCanvas(window.innerWidth,window.innerHeight);
gen_random_color();
const model = tf.sequential();
model.add(tf.layers.dense({inputShape: [4], units: 100}));

}
function draw()
{
background(r,g,b);
textSize(60);
fill(0);
text("black",150,350)
fill(255);
text("white",1000,350)
}

