//Get some important variable by writing them into txt
//var fso=new ActiveXObject("Scripting.FileSystemObject");
//var FILE_write=fso.createtextfile("D:/summer_research/project_0/variable_in_auto.txt",2,true);


//debugger
//WB.AppletList.Applet("DSApplet").App.Script.doToolsRunMacro("C:/Users/student/Desktop/ansysfile/1111.js")

var DS= WB.AppletList.Applet("DSApplet").App.Script.ds;
//var ListView = DS.Script.lv;
var SC=DS.Script;
var SM=SC.sm;

// Clear()  
// This function resets the selection manager so that no geometry is currently 
// selected.


//select all part(bodies)
SC.doGraphicsPartSelect();   //always do part selection
SC.doEditsToolbar(8047);   //select all according to cursor mode


/*
//change the contact type to frictional set the coefficient
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
ListView.ItemValue = "0.03" ;

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


var name_array = new Array();

if (SM.Parts > 0) {
    // Do the real work
    //create_faces_ns_from_body();
	// See structure definition below.
    var face_id_map = new Array(SM.Parts);

    //FILE_write.writeLine("SM.SelectedCount:"+SM.SelectedCount);

    // First we want to iterate over the selected parts and create
    // a list of all of the face ids for each selected part
    for (var i = 1; i <= SM.SelectedCount; i++) {
    
        var topo_id = SM.SelectedEntityTopoID(i);
        var part_id = SM.SelectedPartID(i);
        //FILE_write.writeLine(i+"(topo_id): "+topo_id);
        //FILE_write.writeLine(i+"(part_id): "+part_id);

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

            //FILE_write.writeLine(f + "(faces_ids): " + face_ids);
            //FILE_write.writeLine("faces("+f+").id=" + faces(f).Id);
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
        
        //FILE_write.writeLine("face_id_map["+(i-1)+"][0]:"+face_id_map[i-1][0]);
        //FILE_write.writeLine("face_id_map["+(i-1)+"][1]:"+face_id_map[i-1][1]);
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
            
            //FILE_write.writeLine(j + "(name): " + name);
            name_array[j+i*face_ids.length] = name;
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
ListView.ItemValue = name_array[0];


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



var x = 0;
var y = 0;
var z = 0;

var filespec='D:/XiaoTianyi/Real/pro/2_7_2019_files/dp0/SYS/MECH/file1.rst'; //file for failed solution
var strSourFile = 'D:/XiaoTianyi/Real/pro/2_7_2019_files/dp0/SYS/MECH/file.rst';
var objFSO = new ActiveXObject("Scripting.FileSystemObject");

/*
var fso_r=new ActiveXObject("Scripting.FileSystemObject");
if (fso_r.FileExists("D:/summer_research/Real/x_y_z.txt")){

    var FILE_read=fso_r.OpenTextFile("D:/summer_research/Real/x_y_z.txt");
    real_x = FILE_read.ReadLine();
    real_y = FILE_read.ReadLine();
    real_z = FILE_read.ReadLine();
    FILE_read.close();

    //save file
   
    if(objFSO.FileExists(strSourFile)){
        if(!objFSO.FileExists(filespec)){
            var strDestFile1 = 'D:/summer_research/Real/rst/1_'+real_x+'_'+real_y+'_'+real_z+'.rst';
            var strPath = objFSO.MoveFile(strSourFile, strDestFile1);
        }
        else{
            var strDestFile0 = 'D:/summer_research/Real/rst/0_'+real_x+'_'+real_y+'_'+real_z+'.rst';
            var strPath = objFSO.MoveFile(strSourFile, strDestFile1);
        }
    }
}
*/


for (var i=0;i<20;i++)
{
	var magnitude = Math.random()*450+50;
	
	for (var j=0;j<100;j++)
	{
		var point=randomSpherePoint(0,0,0,magnitude);
		x=point[0];
		y=point[1];
		z=point[2];
		
		//var SelForce = DS.Tree.FirstActiveBranch.Environment.Loads.Item(3+i);
		//DS.Script.changeActiveObject(SelForce.ID);
		
		if (i!=0 || j!=0){
			ListView.ActivateItem("Suppressed");
			ListView.ItemValue="Yes";
		}
		
		//insert force
		DS.Script.doInsertEnvironmentForce();
		ListView.ActivateItem("Scoping Method");
		ListView.ItemValue = "Named Selection" ;
		ListView.ActivateItem("Named Selection");
		ListView.ItemValue = name_array[5] ;
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
		
		if(objFSO.FileExists(strSourFile)){
			if(!objFSO.FileExists(filespec)){
				var strDestFile1 = 'D:/XiaoTianyi/Real/rst/1_'+x+'_'+y+'_'+z+'.rst';
				var strPath = objFSO.MoveFile(strSourFile, strDestFile1);
			}
			else{
				var strDestFile0 = 'D:/XiaoTianyi/Real/rst/0_'+x+'_'+y+'_'+z+'.rst';
				var strPath = objFSO.MoveFile(strSourFile, strDestFile1);
			}
		}


		//var fso_w = new ActiveXObject("Scripting.FileSystemObject");
		//if (!fso_w.FileExists("D:/summer_research/Real/x_y_z.txt")){
		//    var FILE_write = fso_w.CreateTextFile("D:/summer_research/Real/x_y_z.txt",2,true);
		//}
		//else{
		//    var FILE_write = fso_w.OpenTextFile("D:/summer_research/Real/x_y_z.txt",2,true);
		//}
		/*
		var FILE_write = fso_r.OpenTextFile("D:/summer_research/Real/x_y_z.txt",2,true);
		FILE_write.writeLine(x);
		FILE_write.writeLine(y);
		FILE_write.writeLine(z);
		FILE_write.close();
		*/
	}
}



