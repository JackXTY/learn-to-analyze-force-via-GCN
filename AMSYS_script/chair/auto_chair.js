//debugger
//WB.AppletList.Applet("DSApplet").App.Script.doToolsRunMacro("C:/Users/student/Desktop/ansysfile/1111.js")

var DS= WB.AppletList.Applet("DSApplet").App.Script.ds;

//var ListView = DS.Script.lv;

var SC=DS.Script

var SM=SC.sm

// Clear()  
// This function resets the selection manager so that no geometry is currently 
// selected.


//select all part(bodies)
SC.doGraphicsPartSelect()   //always do part selection
SC.doEditsToolbar(8047)   //select all according to cursor mode


//change the contact type to frictional set the coefficient
/*
var c1 = DS.Tree.FirstActiveModel.ContactGroup; 
var pair1 = c1.Children.Item(1); 
var pair2=pair1.Children.Item(1);
pair2.ContactType = SC.id_FrictionalContact; 
pair2.FrictionCoeff = 0.1; 
*/

//Mesh
SC.doModelPreviewMeshSelection();
var Mesh_Mod = DS.Tree.FirstActiveBranch.MeshControlGroup;
SC.SelectItems(" " +Mesh_Mod.ID);
ListView.ActivateItem("Element Size");
ListView.ItemValue = "0.02" ;


//name each face
var basename = 'faces';
var BODY_TOPO_TYPE = 3;
var FACE_TOPO_TYPE = 2;
var EDGE_TOPO_TYPE = 1;
var VERT_TOPO_TYPE = 0;


function classify_entity_type(topo_id) {
    // The top two bits store the topological entity type within the topo
    // id value
    var type = topo_id >>> 30;
    return type;
}

//var fso=new ActiveXObject("Scripting.FileSystemObject");
//var FILE_write=fso.createtextfile("D:/summer_research/Real/assembly/variable_in_auto.txt",2,true);
var name_array = new Array();

if (SM.Parts > 0) {
    // Do the real work
    //create_faces_ns_from_body();
	// See structure definition below.
    var face_id_map = new Array(SM.Parts);

    // First we want to iterate over the selected parts and create
    // a list of all of the face ids for each selected part
    for (var i = 1; i <= SM.SelectedCount; i++) {
        var topo_id = SM.SelectedEntityTopoID(i);
        var part_id = SM.SelectedPartID(i);

        // Make sure this is a body.  If not, just continue around
        if (classify_entity_type(topo_id) != BODY_TOPO_TYPE) {
            continue;
        }
        var part = SM.PartMgr.PartById(part_id);
        var brep = part.BRep;
        // Pull the body object out of the BRep structure.  The call
        // for this is the Cell function with the topological id for
        // the body of interest
        body = brep.Cell(topo_id);
                
        // This array will be used to hold a list of face ids for a given
        // body.
        var face_ids = new Array();
        // These are the actual face objects associated with this body
        var faces = body.Faces;
        // Now store all of the face ids in our face id list
        for (var f = 1; f <= faces.Count; f++) {
            face_ids.push(faces(f).Id);
        }
        // Now that we have the face ids, put them into an array of structures
        // that each look like the following:
        // |-------|
        // |   0   |-> Part ID
        // |-------|
        // |-------|
        // |   1   |-> List of face ids for this part ID
        // |-------|
        face_id_map[i - 1] = new Array(2);
        face_id_map[i - 1][0] = part_id;
        face_id_map[i - 1][1] =
            face_ids.slice(0, face_ids.length); // Slice creates a copy of the array

    }
    // Now that we've built up our datastructure of face ids, we need to select them all
    SM.Clear();
    var name = null;
    for (var i = 0; i < face_id_map.length; i++) {
        var part_id = face_id_map[i][0];
        var face_ids = face_id_map[i][1];
        for (var j = 0; j < face_ids.length; j++) {
            SM.Clear();
            // Create a unique name based on the part id and face id
            name = basename + '_' + part_id.toString() + '_' + face_ids[j].toString();
            SM.AddToSelection(part_id, face_ids[j], false);
            // Create the component
            SC.addNamedSelection(false, name, SC.id_NS_UnknownMultiCriterion);      
            
            name_array[j+i*face_ids.length] = name;

            //FILE_write.writeLine(j + "(name): " + name);
        }
    }
	//SM.Clear();
} else {
    SC.WBScript.Out(SM.Parts, true);
}


var Env = DS.Tree.FirstActiveBranch.Environment;
SC.SelectItems(" " +Env.ID);



//insert gravity
SC.doInsertEnvironmentGravity(1);


//insert fixed support
SC.doInsertEnvironmentFixedDisplacement();
ListView.ActivateItem("Scoping Method");
ListView.ItemValue = "Named Selection" ;
ListView.ActivateItem("Named Selection");
ListView.ItemValue = "faces_3_2147483650";

SC.doInsertEnvironmentFixedDisplacement();
ListView.ActivateItem("Scoping Method");
ListView.ItemValue = "Named Selection" ;
ListView.ActivateItem("Named Selection");
ListView.ItemValue = "faces_3_2147483654";

/*
//insert force
DS.Script.doInsertEnvironmentForce();
ListView.ActivateItem("Scoping Method");
ListView.ItemValue = "Named Selection" ;
ListView.ActivateItem("Named Selection");
ListView.ItemValue = "faces_3_2147483654" ;
ListView.ActivateItem("Define By");
ListView.ItemValue = "Components" ;
ListView.ActivateItem("X Component");
ListView.ItemValue = "0";
ListView.SelectedItem.IsChecked="true";
ListView.ActivateItem("Y Component");
ListView.ItemValue = "0";
ListView.SelectedItem.IsChecked="true";
ListView.ActivateItem("Z Component");
ListView.ItemValue = "5";
ListView.SelectedItem.IsChecked="true";
*/

function randomSpherePoint(x0,y0,z0,radius){
   var u = Math.random();
   var v = Math.random();
   var theta = 2 * Math.PI * u;
   var phi = Math.acos(2 * v - 1);
   var x = x0 + (radius * Math.sin(phi) * Math.cos(theta));
   var y = y0 + (radius * Math.sin(phi) * Math.sin(theta));
   var z = z0 + (radius * Math.cos(phi));
   return [x,y,z];
}




for (var i=0;i<995;i++)
{
    var force_magnitude = Math.random()*300+60;

    for(var j=0;j<2;j++)
    {
        var point=randomSpherePoint(0,0,0,force_magnitude);
        var x=point[0];
        var y=point[1];
        var z=point[2];
        
        //var SelForce = DS.Tree.FirstActiveBranch.Environment.Loads.Item(3+i);
        //DS.Script.changeActiveObject(SelForce.ID);
        if(i!=0||j!=0){
            ListView.ActivateItem("Suppressed");
            ListView.ItemValue="Yes";
        }
        DS.Script.doInsertEnvironmentForce();
        ListView.ActivateItem("Scoping Method");
        ListView.ItemValue = "Named Selection" ;
        ListView.ActivateItem("Named Selection");
        ListView.ItemValue = "faces_3_2147483656" ;
        ListView.ActivateItem("Define By");
        ListView.ItemValue = "Components" ;
        ListView.ActivateItem("X Component");
        ListView.ItemValue = x;
        ListView.SelectedItem.IsChecked="true";
        ListView.ActivateItem("Y Component");
        ListView.ItemValue = y;
        ListView.SelectedItem.IsChecked="true";
        ListView.ActivateItem("Z Component");
        ListView.ItemValue = z;
        ListView.SelectedItem.IsChecked="true";


        
        //solve
        SC.doSolveDefaultHandler();
        //save file
        var filespec='D:/XiaoTianyi/Real/Model2_chair/pro/9_8_2019_files/dp0/SYS/MECH/file1.rst';
        var strSourFile = 'D:/XiaoTianyi/Real/Model2_chair/pro/9_8_2019_files/dp0/SYS/MECH/file.rst';
        
        var objFSO = new ActiveXObject("Scripting.FileSystemObject");
        if(!objFSO.FileExists(filespec)){
            var strDestFile = 'D:/XiaoTianyi/Real/Model2_chair/rst/1_'+x+'_'+y+'_'+z+'.rst';
            var strPath = objFSO.CopyFile(strSourFile, strDestFile);
        }
        else{
            var strDestFile = 'D:/XiaoTianyi/Real/Model2_chair/rst/0_'+x+'_'+y+'_'+z+'.rst';
            var strPath = objFSO.CopyFile(strSourFile, strDestFile);
        }
    }
}



