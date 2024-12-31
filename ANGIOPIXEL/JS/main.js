// rest.get(url, callback)
// rest.post(url, body, callback)
// rest.put(url, body, callback)
// rest.delete(url, callback)
// function callback(estado, respuesta) {...}

const res = require("express/lib/response");

/* Funcion acceder: Actua cuando se pulsa el botón Entrar de "access". Envía los datos del usuario y contraseña al
servidor para confirmar los datos. Pasa de la ventana "access" a "start", da un mensaje de bienvenida con el
nombre del medico, guarda la id del medico, dibuja la tabla con los expedientes correspondientes con el medico.
Requiere: null. Devuelve: "null".
*/
function acceder(){
  var user = document.getElementById("user").value;
  var password = document.getElementById("pass1").value;
  var log = {usuario: user, clave: password};
  var id_user;
  console.log("llego a enviar");
  rest.post("/user/login",log,function(estado,resp){
    if (estado == 403){
      alert(resp);
      return;
    }
    localStorage.setItem("token", resp);
    mover('sesion.html');
  })
}

function mover(direccion){
  location.href = direccion;
}
/* Funcion guardar_reg: Actua cuando se pulsa el botón Guardar de "medical_data". Envía los datos del registro del
 medico al servidor para que se guarden. Pasa de la ventana "medical_data" a "access", autocompleta el nombre de
 usuario y contraseña que se hubiera colocado en "medical_data". Requiere: null. Devuelve: "null"
*/
function guardar_reg(){
  var usuario = document.getElementById("user");
  var contra = document.getElementById("pass1");
  var name = document.getElementById("name1").value;
  var lastn = document.getElementById("lastname1").value;
  var user = document.getElementById("login").value;
  var pass = document.getElementById("pass2").value;
  var cent = document.getElementById("center").value;

  var usuario = {nombre: name, apellidos: lastn, login:user, password: pass, centro: cent};
  //Se envía los datos

  rest.post("/user/register",usuario,function(estado,mensaje){
    if (estado != 201){
      alert(mensaje);
      return;
    }
    else{
      alert(mensaje);
      mover("index.html");
    }
  })
}
/* Crea un enlace temporal con la finalidad de que el usuario pueda extraer sus resultados */
function exportar_reg() {
  // Crea un contenido CSV
  let contenidoCSV = 'Nombre,CNN,SVM,YOLO\n'; // Encabezados del archivo CSV

  for (var archivo of archivos) {
    contenidoCSV += `${archivo.nombre},${archivo.CNN},${archivo.SVM},${archivo.YOLO}\n`;
  }

  // Crea un Blob con el contenido CSV
  const blob = new Blob([contenidoCSV], { type: 'text/csv' });

  // Genera una URL para el Blob
  const url = URL.createObjectURL(blob);

  // Crea un elemento <a> para la descarga
  const enlace = document.createElement('a');
  enlace.href = url;
  enlace.download = 'registro_modelo.csv'; // Nombre del archivo
  enlace.setAttribute('style', 'display: none');

  // Agrega el enlace al DOM, simula un clic y lo elimina
  document.body.appendChild(enlace);
  enlace.click();
  document.body.removeChild(enlace);

  // Libera la URL para optimizar recursos
  URL.revokeObjectURL(url);
}

function exportar_img(){
  const elem_img = document.getElementById("img123");
  var nombre_img = elem_img.getAttribute("class");
  var url_img = elem_img.getAttribute("src");
  descargarImagenDesdeURL(url_img,nombre_img);
}
/* Descarga de la img por url */
function descargarImagenDesdeURL(url, nombreArchivo) {
  // Crear una nueva imagen
  const img = new Image();
  img.crossOrigin = 'anonymous'; // Evita problemas de CORS

  // Definir una acción cuando la imagen cargue
  img.onload = () => {
      // Crear un canvas del tamaño de la imagen
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;

      // Dibujar la imagen en el canvas
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);

      // Convertir el contenido del canvas a una URL de datos
      const urlDescarga = canvas.toDataURL('image/png');

      // Crear un enlace para descargar
      const enlace = document.createElement('a');
      enlace.href = urlDescarga;
      enlace.download = nombreArchivo || 'imagen.png';
      
      // Simular un clic en el enlace
      document.body.appendChild(enlace);
      enlace.click();
      document.body.removeChild(enlace);
  };

  // Manejar errores
  img.onerror = () => {
      console.error('Error al cargar la imagen desde la URL');
  };

  // Establecer la URL de la imagen
  img.src = url;
}

async function descargarImagenesComoZip() {
  var listaImagenes = archivos;
  // Verifica que JSZip esté disponible
  if (typeof JSZip === 'undefined') {
    console.error('JSZip no está disponible. Asegúrate de incluir la biblioteca JSZip.');
    return;
  }

  const zip = new JSZip(); // Crea una nueva instancia de JSZip
  const carpeta = zip.folder('imagenes'); // Crea una carpeta dentro del ZIP

  for (const item of listaImagenes) {
    try {
      const respuesta = await fetch(item.url); // Descarga la imagen
      const blob = await respuesta.blob(); // Convierte la respuesta en un Blob
      carpeta.file(item.nombre, blob); // Agrega el archivo al ZIP
    } catch (error) {
      console.error(`Error al descargar la imagen: ${item.url}`, error);
    }
  }

  // Genera el archivo ZIP y descarga
  zip.generateAsync({ type: 'blob' }).then((contenidoZip) => {
    const enlace = document.createElement('a');
    enlace.href = URL.createObjectURL(contenidoZip);
    enlace.download = 'imagenes.zip';
    enlace.style.display = 'none';
    document.body.appendChild(enlace);
    enlace.click();
    document.body.removeChild(enlace);
  });
}


/* Aplicar toma los datos actuales en pantalla (nombre img, filtros y modelos a aplicar) y los añade a var mensajes.
Se puede ver desde consola.
*/
function aplicar() {
  const imagen = document.getElementById("img123");
  var modelos = getModelos();
  //Si el user quiere rectificar puede hacerlo
  var mensajes_length = mensajes.length - 1;
  for(var i = mensajes_length; i > -1;i--){
    if(imagen.getAttribute("class")==mensajes[i].img){
      mensajes.splice(i,1);
    }
  }
  //
  for(var file of archivos){
    if(imagen.getAttribute("class")==file.nombre){
      for(var modelo of modelos){
        mensajes.push({img:file.nombre,modelo:modelo});
      }
    }
  }
  alert("Los cambios han sido guardados");
}
function aplicar_todos(){
  var modelos = getModelos();
  var mensajes_length = mensajes.length - 1;
  for(var i = mensajes_length; i > -1;i--){
    mensajes.splice(i,1);
  }
  for(var file of archivos){
    for(var modelo of modelos){
      mensajes.push({img:file.nombre,modelo:modelo});
    }
  }
  alert("Los cambios han sido guardados");
}
/* Obtiene el valor de los filtros segun el estado checked*/
function getFiltros(){
  const filtro1 = document.getElementById("check-filtro1");
  const filtro2 = document.getElementById("check-filtro2");
  const filtro3 = document.getElementById("check-filtro3");
  const filtro4 = document.getElementById("check-filtro4");
  const filtros = [];
  if(filtro1.checked==true){filtros.push(filtro1.value);}
  if(filtro2.checked==true){filtros.push(filtro2.value);}
  if(filtro3.checked==true){filtros.push(filtro3.value);}
  if(filtro4.checked==true){filtros.push(filtro4.value);}
  return filtros
}
/* Obtiene el valor de los modelos segun el estado checked*/
function getModelos(){
  const modelo1 = document.getElementById("check-modelo1");
  const modelo2 = document.getElementById("check-modelo2");
  const modelo3 = document.getElementById("check-modelo3");
  //const modelo4 = document.getElementById("check-modelo4");
  const modelos = [];
  if(modelo1.checked==true){modelos.push(modelo1.value);}
  if(modelo2.checked==true){modelos.push(modelo2.value);}
  if(modelo3.checked==true){modelos.push(modelo3.value);}
  //if(modelo4.checked==true){modelos.push(modelo4.value);}
  return modelos;
}
/* Se envia al servidor la lista de mensajes.*/
function enviar(posicion){
  alert("Esta operación puede tardar unos minutos. Espere por favor");
  var i = 0;
  let prediccion = "";
  for(var mensaje of mensajes){//Hace una llamada al servidor tantas veces como modelos e img tiene que cargar
    llamada_py(mensaje,function(res){
      i = res.result.indexOf("Modelo");
      prediccion = res.result.substr(i);
      for(var p = (archivos.length-1); p > -1;p--){
        if(res.img==archivos[p].nombre){
          if(res.result.indexOf("CNN")>-1){archivos[p].CNN = prediccion;}
          if(res.result.indexOf("SVM")>-1){archivos[p].SVM = prediccion;}
          if(res.result.indexOf("yolo")>-1){archivos[p].YOLO = prediccion;}
          if(res.result.indexOf("angionet")>-1){archivos[p].Angionet = prediccion;}//
        }
      }
      mostrar_img(posicion);
    });
  }
}
async function llamada_py(enviar,cb){
  let result = "";
  var cuerpo = JSON.stringify(enviar);
  try {
    const response = await fetch("/user/prueba", {
      method: "POST",
      headers: {
        "Content-Type": "application/json; charset=utf-8",
      },
      body: cuerpo,
    });

    if (!response.ok) {
      const error = await response.text();
      console.error("Error del servidor:", error);
      return;
    }

    const data = await response.json();
    console.log("Resultado del servidor:", data.result.txt);
    result += data.result.txt;
    cb({img:data.result.img, result:result});
  } catch (error) {
    console.error("Error en la solicitud:", error);
  }
}

function mostrar_filtro(){
  var filtros = getFiltros();
  var elem_img = document.getElementById("img123");
  var url = elem_img.getAttribute("src");
  var nombre = elem_img.getAttribute("class");
  uploadFile(url,nombre);
  console.log("Se ha guardado la img");
  llamada_filtros(nombre,filtros,function(resp){
    console.log(resp);
    obtener_img(nombre);
  });
}
function mostrar_filtro_todos(){
  var filtros = getFiltros();
  for(var archivo of archivos){
    uploadFile(archivo.url,archivo.nombre);
  }
  for(var archivo of archivos){
    var nombre = archivo.nombre;
    var i = 0;
    llamada_filtros(nombre,filtros,function(resp){
      obtener_img(archivos[i].nombre);
      i++;
    })
  }
}
function llamada_filtros(nombre,body,cb){
  console.log("se hace la llamada a filtro");
  body = JSON.stringify(body);
  rest.post("user/filtrar/"+nombre,body,function(estado,resp){
    if(estado==500){
      console.log("Ha habido un error en el servidor");
    }
    cb(resp); // Se supone que debería darme un file img_filtro;
  });
}
function obtener_img(nombre) {
  fetch(`/user/obtener/${nombre}`, { method: 'POST' })
      .then(response => {
          if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
          }
          return response.blob(); // Convertir la respuesta en un Blob
      })
      .then(blob => {
          // Crear una URL para el blob recibido
          const fileUrl = URL.createObjectURL(blob);

          // Asignar la URL generada al objeto correspondiente en 'archivos'
          for (var archivo of archivos) {
              if (archivo.nombre === nombre) {
                  archivo.url = fileUrl;
                  var elem_img = document.getElementById("img123");
                  for(var archivo of archivos){
                    if(archivo.nombre == elem_img.getAttribute("class")){
                      elem_img.setAttribute("src",archivo.url);
                    }
                  }
              }
          }
      })
      .catch(error => {
          console.error("Error al obtener la imagen:", error);
      });
}



function revertir_img(posicion){
  var elem_revertir_img = document.getElementById("btn-revertir-img");
  var url = elem_revertir_img.getAttribute("src");
  archivos[posicion].url=url;
  mostrar_img(posicion);
}
function revertir_img_todos(posicion){
  for(var archivo of archivos){
    archivo.url=archivo.org_url;
  }
  mostrar_img(posicion);
}
function revertir_modelos(){
  var modelos = getModelos();
  var mensajes_length = mensajes.length - 1;
  for(var i = mensajes_length; i > -1;i--){
    mensajes.splice(i,1);
  }
  alert("Desechos todos los cambios");
}

function filtrar(){
  mostrar_img(0);
  const subida = document.getElementById("subida");
  const filtrado = document.getElementById("filtrado");
  const b_sig = document.getElementById("btn-der");
  const b_ret = document.getElementById("btn-izq");
  subida.setAttribute("style","display: none");
  filtrado.setAttribute("style","display: block");
  b_sig.setAttribute("onclick","avance("+1+")");
  b_ret.setAttribute("onclick","avance()");
  for(var file of archivos){
    uploadFile(file.url,file.nombre);//Si la img ya existe (nombre del archivo ya esta dentro de /ANGIOPIXEL/Local/) el servidor sobreescribe el archivo
  }
}
/* Muestra por pantalla la img cuya posicion de lista archivos pasa por parametro */
function mostrar_img(posicion){
  var elem_img = document.getElementById("img123");
  elem_img.setAttribute("src",archivos[posicion].url);
  elem_img.setAttribute("class",archivos[posicion].nombre);
  var elem_CNN = document.getElementById("resultadoCNN");
  var elem_SVM = document.getElementById("resultadoSVM");
  var elem_YOLO = document.getElementById("resultadoYolo");
  var elem_Angionet = document.getElementById("resultadoAngionet");
  elem_CNN.innerText=archivos[posicion].CNN;
  elem_SVM.innerText=archivos[posicion].SVM;
  elem_YOLO.innerText=archivos[posicion].YOLO;
  //elem_Angionet.innerText="Resultado CNN\n"+archivos[posicion].Angionet;
  var elem_enviar = document.getElementById("btn-enviar");
  elem_enviar.setAttribute("onclick","enviar("+posicion+")");
  var elem_revertir_img = document.getElementById("btn-revertir-img");
  var elem_revertir_img_todos = document.getElementById("btn-revertir-img-todos");
  elem_revertir_img.setAttribute("src",archivos[posicion].org_url);
  elem_revertir_img.setAttribute("onclick","revertir_img("+posicion+")");
  elem_revertir_img_todos.setAttribute("onclick","revertir_img_todos("+posicion+")");
}
//hacer botones de delante y atras
function avance(num){
  const btnIzq = document.getElementById("btn-izq");
  const btnDer = document.getElementById("btn-der");
  if((num-1) > -1){
    btnIzq.setAttribute("onclick","avance("+(num-1)+")");
    mostrar_img(num);
  }
  if((num+1) < archivos.length){
    btnDer.setAttribute("onclick","avance("+(num+1)+")");
    mostrar_img(num);
  }
}

function tokenSesion(){
  var item = localStorage.getItem("token");
  if(item==null){
    window.location='/index.html';
  }
}