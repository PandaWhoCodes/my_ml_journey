
var r,g,b;
var ai;
var locations = [[150,350],[1000,350]];
var training_data = [];
var output_data = [];
var count = 0;
var net = new brain.NeuralNetwork();

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
    draw_location(0,1);
    draw_location(1,0);
}

function draw_location(loc,choice)
{
    textAlign(CENTER,CENTER);
    var text1 = (loc==0) ? "BLACK\n" : "WHITE";
    if (choice == 1)
    {
        text1 = text1 + choice;
    }
    textSize(50);
    fill(loc*255);
    text(text1,locations[loc][0],locations[loc][1])
}

function keyPressed() {


    var op;
    if (keyCode === RIGHT_ARROW) {
        op = "black";

    }
    else if (keyCode === LEFT_ARROW) {
        op = "white";
    }
    training_data.push({ "rr": r, "gg": g, "bb": b });
    output_data.push({ "op": op });
    count++;
    var to_train = get_formatted();
    console.log(to_train);
    net.train(
    to_train
     );
    var opp=net.run({ "rr": r, "gg": g, "bb": b });
    console.log(opp);
  }
function get_formatted()
{
    var return_value =[];
    for(var i=0;i<count;i++)
    {
        return_value.push({input: training_data[i],output:output_data[i]})
    }
    return return_value;
}