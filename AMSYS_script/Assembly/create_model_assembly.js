var Yes = agc.Yes;
var No  = agc.No;



//creat the first sketch
function planeSketchesYZOnly (p)
{

//Plane
p.Plane  = agb.GetYZPlane();
p.Origin = p.Plane.GetOrigin();
p.XAxis  = p.Plane.GetXAxis();
p.YAxis  = p.Plane.GetYAxis();

//Sketch
p.Sk2 = p.Plane.NewSketch();
p.Sk2.Name = "Sketch2";

//Edges
with (p.Sk2)
{
  p.Ln17 = Line(-0.15000000+0.05, 0.05000000+0.05, -0.05000000+0.05, 0.05000000+0.05);
  p.Ln18 = Line(-0.05000000+0.05, 0.05000000+0.05, -0.05000000+0.05, 0.00000000+0.05);
  p.Ln19 = Line(-0.05000000+0.05, 0.00000000+0.050, 0.05000000+0.05, 0.00000000+0.05);
  p.Ln20 = Line(0.05000000+0.05, 0.00000000+0.05, 0.05000000+0.05, 0.05000000+0.05);

  p.Ln21 = Line(0.05000000+0.05, 0.05000000+0.05, 0.15000000+0.05, 0.05000000+0.05);
  p.Ln22 = Line(0.15000000+0.05, 0.05000000+0.05, 0.15000000+0.05, -0.05000000+0.05);
  p.Ln23 = Line(0.15000000+0.05, -0.05000000+0.05, -0.15000000+0.05, -0.05000000+0.05);
  p.Ln24 = Line(-0.15000000+0.05, -0.05000000+0.05, -0.15000000+0.05, 0.05000000+0.05);
}

//Dimensions and/or constraints
with (p.Plane)
{
	/*
  //Dimensions
  var dim;
  dim = HorizontalDim(p.Ln10.End, -0.15000000, 0.05000000, 
    p.Ln8.Base, 0.15000000, 0.05000000, 
    0.07269939, 0.20956459);
  if(dim) dim.Name = "H1";
  dim = HorizontalDim(p.Origin, 0.00000000, 0.00000000, 
    p.Ln10.End, -0.15000000, 0.05000000, 
    -0.09527027, 0.12857347);
  if(dim) dim.Name = "H2";
  dim = VerticalDim(p.Ln9.End, -0.15000000, -0.05000000, 
    p.Ln7.Base, -0.15000000, 0.05000000, 
    -0.35055857, 0.00789474);
  if(dim) dim.Name = "V3";
  dim = VerticalDim(p.Origin, 0.00000000, 0.00000000, 
    p.Ln7.Base, -0.15000000, 0.05000000, 
    -0.21154397, 0.02758241);
  if(dim) dim.Name = "V4";
  */

  //Constraints
  HorizontalCon(p.Ln17);
  HorizontalCon(p.Ln19);
  HorizontalCon(p.Ln21);
  HorizontalCon(p.Ln23);
  VerticalCon(p.Ln18);
  VerticalCon(p.Ln20);
  VerticalCon(p.Ln22);
  VerticalCon(p.Ln24);
  CoincidentCon(p.Ln17.End, -0.05000000+0.05, 0.05000000+0.05, 
                p.Ln18.Base, -0.05000000+0.05, 0.05000000+0.05);
  CoincidentCon(p.Ln18.End, -0.05000000+0.05, 0.00000000+0.05, 
                p.Ln19.Base, -0.05000000+0.05, 0.00000000+0.05);
  CoincidentCon(p.Ln19.End, 0.05000000+0.05, 0.00000000+0.05, 
                p.Ln20.Base, 0.05000000+0.05, 0.0000000+0.050);
  CoincidentCon(p.Ln20.End, 0.05000000+0.05, 0.05000000+0.05, 
                p.Ln21.Base, 0.05000000+0.05, 0.05000000+0.05);
				
  CoincidentCon(p.Ln21.End, 0.15000000+0.05, 0.05000000+0.05, 
				p.Ln22.Base, 0.15000000+0.05, 0.05000000+0.05);
  CoincidentCon(p.Ln22.End, 0.15000000+0.05, -0.05000000+0.05, 
				p.Ln23.Base, 0.15000000+0.05, -0.05000000+0.05);
  CoincidentCon(p.Ln23.End, -0.15000000+0.05, -0.05000000+0.05, 
				p.Ln24.Base, -0.15000000+0.05, -0.05000000+0.05);
  CoincidentCon(p.Ln24.End, -0.15000000+0.05, 0.05000000+0.05, 
				p.Ln17.Base, -0.15000000+0.05, 0.05000000+0.05);
}

p.Plane.EvalDimCons(); //Final evaluate of all dimensions and constraints in plane

return p;
} //End Plane JScript function: planeSketchesOnly

//Call Plane JScript function
var ps2 = planeSketchesYZOnly (new Object());

//Finish
agb.Regen(); //To insure model validity




/*
//create a new plane to create the second sketch
var plyz=agb.GetYZPlane();
var pl4=agb.PlaneFromPlane(plyz);
if(pl4)
{
	pl4.Name="Plane4";
    pl4.ReverseNormal = No;
    pl4.ReverseAxes = No;
    pl4.ExportCS = No;
	pl4.AddTransform(agc.XformZOffset, -0,05);
}
agb.Regen();

//create the second sketch
agb.SetActivePlane(pl4);
*/

//creat the first sketch
function planeSketchesZXOnly (p)
{

//Plane

//p.Plane = agb.GetActivePlane();
p.Plane  = agb.GetZXPlane();
p.Origin = p.Plane.GetOrigin();
p.XAxis  = p.Plane.GetXAxis();
p.YAxis  = p.Plane.GetYAxis();

//Sketch
p.Sk1 = p.Plane.NewSketch();
p.Sk1.Name = "Sketch1";

//Edges
with (p.Sk1)
{
 
  p.Ln7 = Line(0, 0, 0.05, 0);
  p.Ln8 = Line(0.05, 0, 0.05, 0.1);
  p.Ln9 = Line(0.05, 0.1, 0, 0.1);
  p.Ln10 = Line(0, 0.1, 0, 0.2);
  
  p.Ln11 = Line(0, 0.2, 0.1, 0.2);
  p.Ln12 = Line(0.1, 0.2, 0.1, -0.1);
  p.Ln13 = Line(0.1, -0.1, 0, -0.1);
  p.Ln14 = Line(0, -0.1, 0, 0);
}

//Dimensions and/or constraints
with (p.Plane)
{
	/*
  //Dimensions
  var dim;
  dim = HorizontalDim(p.Ln10.End, -0.15000000, 0.05000000, 
    p.Ln8.Base, 0.15000000, 0.05000000, 
    0.07269939, 0.20956459);
  if(dim) dim.Name = "H1";
  dim = HorizontalDim(p.Origin, 0.00000000, 0.00000000, 
    p.Ln10.End, -0.15000000, 0.05000000, 
    -0.09527027, 0.12857347);
  if(dim) dim.Name = "H2";
  dim = VerticalDim(p.Ln9.End, -0.15000000, -0.05000000, 
    p.Ln7.Base, -0.15000000, 0.05000000, 
    -0.35055857, 0.00789474);
  if(dim) dim.Name = "V3";
  dim = VerticalDim(p.Origin, 0.00000000, 0.00000000, 
    p.Ln7.Base, -0.15000000, 0.05000000, 
    -0.21154397, 0.02758241);
  if(dim) dim.Name = "V4";
  */

  //Constraints
  HorizontalCon(p.Ln7);
  HorizontalCon(p.Ln9);
  HorizontalCon(p.Ln11);
  HorizontalCon(p.Ln13);
  VerticalCon(p.Ln8);
  VerticalCon(p.Ln10);
  VerticalCon(p.Ln12);
  VerticalCon(p.Ln14);
  CoincidentCon(p.Ln7.End, 0.05, 0, 
                p.Ln8.Base, 0.05, 0);
  CoincidentCon(p.Ln8.End, 0.05, 0.1, 
                p.Ln9.Base, 0.05, 0.1);
  CoincidentCon(p.Ln9.End, 0, 0.1, 
                p.Ln10.Base, 0, 0.1);
  CoincidentCon(p.Ln10.End, 0, 0.2, 
                p.Ln11.Base, 0, 0.2);
				
  CoincidentCon(p.Ln11.End, 0.1, 0.2, 
				p.Ln12.Base,  0.1, 0.2);
  CoincidentCon(p.Ln12.End,  0.1, -0.1, 
				p.Ln13.Base, 0.1, -0.1);
  CoincidentCon(p.Ln13.End,  0, -0.1, 
				p.Ln14.Base, 0, -0.1);
  CoincidentCon(p.Ln14.End,  0, 0, 
				p.Ln7.Base, 0, 0);
}

p.Plane.EvalDimCons(); //Final evaluate of all dimensions and constraints in plane

return p;
} //End Plane JScript function: planeSketchesOnly

//Call Plane JScript function
var ps1 = planeSketchesZXOnly (new Object());

//Finish
agb.Regen(); //To insure model validity




//create two bodies
var ext1=agb.Extrude(agc.Frozen,ps1.Sk1,agc.DirNormal,agc.ExtentFixed,0.10,agc.ExtentFixed, 0.0,No,0,0);
agb.Regen();

var ext2=agb.Extrude(agc.Frozen,ps2.Sk2,agc.DirNormal,agc.ExtentFixed,0.10,agc.ExtentFixed, 0.0,No,0,0);
agb.Regen();

//var ext3=agb.Extrude(agc.Add,ps1.Sk1,agc.DirNormal,agc.ExtentFixed,0.10,agc.ExtentFixed, 0.0,No,0,0);
//agb.Regen();





//var boolean1=ag.gui.CreateBoolean();

