document.addEventListener("DOMContentLoaded", () => {
  const cookieBanner = document.getElementById("cookie-consent");
  const acceptButton = document.getElementById("accept-cookies");

  // Mostrar el banner solo si no se ha aceptado previamente
  if (!localStorage.getItem("cookiesAccepted")) {
      cookieBanner.style.display = "block";
  }

  // Manejar el clic en el botón de aceptar
  acceptButton.addEventListener("click", () => {
      localStorage.setItem("cookiesAccepted", "true"); // Guardar estado en localStorage
      cookieBanner.style.display = "none"; // Ocultar el banner
  });
});
