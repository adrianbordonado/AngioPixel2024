var express = require("express");
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { exec } = require("child_process");
const { spawn } = require("child_process");


var app = express();
// Habilitar CORS
app.use(cors()); // Permite solicitudes desde cualquier origen

app.use("/", express.static("ANGIOPIXEL"));

app.use(express.json());

var datos=require("./datos.js");
var usuarios=datos.usrs;

/*Obtiene el obj con el usuario y la contraseña, con un condicional comprueba que es igual a alguno que tenga en la
lista de médicos y devuelve el id si es el caso. De lo contrario devuelve el error 403.*/
app.post("/user/login",function(req, res){
    for (i=0;i<usuarios.length;i++){
        if (usuarios[i].login == req.body.usuario && usuarios[i].password == req.body.clave){
            res.status(200).json(usuarios[i].id);
            return;
        }
    }
    res.status(403).json("Usuario o contraseña incorrectos");
})

/* Obtiene el obj médico, junto con la lista de médicos se introduce en la función incluir() y esta devuelve un
código. Si es el usuario ya existe en la base de datos se devuelve un 200, de lo contrario un 201.*/
app.post("/user/register",function(req,res){
    var f = incluir(req.body,usuarios);
    if( f == 200){
        res.status(f).json("El usuario ya existe");
    }
    else if (f == 201){
        res.status(f).json("El usuario ha sido creado");
    }
})
//---Subida de imágenes---//
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, './ANGIOPIXEL/Local'); // Carpeta donde se almacenarán las imágenes
    },
    filename: (req, file, cb) => {
        cb(null, `${file.originalname}`);
    }
});

const upload = multer({ storage });

app.post("/user/upload",upload.single('file'),function(req,res){
    if(!req.file){
        return res.status(400).send({error:'No se han subido los archivos'});
    }
    res.status(200).send({ message: 'File uploaded successfully', filePath: `/ANGIOPIXEL/Local/${req.file.filename}` });
})

//--Ejecutar modelo en python--//

app.post("/user/prueba",function(req,res){
    //recibe {img:nombre,modelo:modelo,filtros:filtros}
    var mensaje = req.body;
    console.log("Peticion de modelo");
    modelizar(mensaje.modelo,'ANGIOPIXEL/Local/'+mensaje.img,function(cb){
        if (cb == -1) {
            return res.status(500).send("Error procesando la imagen.");
        }
        res.json({ message: "Procesado exitosamente", result: {img:mensaje.img, txt:cb}});
    });
})

//--Ejecutar filtro en python--//

app.post("/user/filtrar/:nombre",function(req,res){
    //recibe [fitros]
    // Ruta de la carpeta donde están almacenadas las imágenes
    const IMAGE_PATH = path.join(__dirname, 'ANGIOPIXEL/Local');
    var mensaje = req.body;
    var nombre=req.params['nombre'];
    // Ruta completa del archivo
    const filePath = path.join(IMAGE_PATH, nombre);
    var filtros = mensaje;
    var y = 0;
    console.log("Peticion de filtros");
    console.log(nombre,filtros);
    var log = "";
    for(var filtro of filtros){
        filtrar(nombre,filtro,function(cb){
            console.log("despues de filtrar");
            if (cb == -1) {
                return res.status(500).json("Error al filtrar la imagen.");
            }
            log += cb;
            y++;
            if(filtros.length==y){
                console.log("Te envio ok");
                res.status(300).json(cb);
            }
        });
    }
})

// Ruta base de las imágenes
const IMAGE_PATH = path.join(__dirname, 'ANGIOPIXEL/Local');

// Endpoint para obtener una imagen por nombre
app.post("/user/obtener/:nombre", function (req, res) {
    const nombre = req.params.nombre;

    // Ruta completa del archivo
    const filePath = path.join(IMAGE_PATH, nombre);

    // Comprobar si el archivo existe
    if (!fs.existsSync(filePath)) {
        return res.status(404).json({ error: `File '${nombre}' not found` });
    }

    // Enviar el archivo al cliente
    res.sendFile(filePath, { headers: { 'Content-Type': 'image/png' } }, (err) => {
        if (err) {
            console.error('Error sending file:', err);
            res.status(500).json({ error: 'Error sending file' });
        }
    });
});

function filtrar(nombre,filtro,cb){
    console.log("Comienza a filtrar..")
    const ruta = 'ANGIOPIXEL/Local/'+ nombre;
    const pythonProcess = spawn("python", [filtro, ruta]);
    var result = "";
    pythonProcess.stdout.on(`data`, (data) => {
        result += data.toString();
    });

    pythonProcess.stderr.on(`data`, (data) => {
        console.error(`Error: ${data.toString()}`);
    });

    pythonProcess.on(`error`, (error) => {
        console.log(`error: ${error}`);
        cb(error);
    });

    pythonProcess.on(`exit`, (code, signal) => {
        if (code) console.log(`Proceso termino con: ${code}`);
        if (signal) console.log(`Proceso kill con: ${signal}`);
        cb(nombre);
    });
}

//----------FUNCIONES---------//
function asigID(lista){
    var id = 0;
    for (i=0;i<lista.length;i++){
        if (id <= lista[i].id){
            id = lista[i].id + 1;//Problemas si se borra un elemento y se pierde una id en el proceso
        }
    }
    return (id);
}

function incluir(objeto,lista){
    for (i=0;i<lista.length;i++){
        if (lista[i].login == objeto.login){
            return (200);
        }
    }
    var o1 = Object.assign({id: asigID(lista)}, objeto);
    usuarios.push(o1);//problema, si se apaga el servidor todo se pierde
    return (201);
}

function modelizar(modelo,ruta,cb){
    const pythonProcess = spawn("python", [modelo, ruta]);
    let result = "";
    pythonProcess.stdout.on(`data`, (data) => {
        result += data.toString();
    });

    pythonProcess.stderr.on(`data`, (data) => {
        console.error(`Error: ${data.toString()}`);
    });

    pythonProcess.on(`error`, (error) => {
        console.log(`error: ${error}`);
        cb(error);
    });

    pythonProcess.on(`exit`, (code, signal) => {
        if (code) console.log(`Proceso termino con: ${code}`);
        if (signal) console.log(`Proceso kill con: ${signal}`);
        cb(result.trim());
        return result.trim();
    });
}

app.listen(3000);
console.log("Servidor activado. Esta escuchando en el puerto: 3000");