document.addEventListener("DOMContentLoaded", () => {

    const form = document.querySelector(".search-box");
    const input = document.querySelector("input[name='question']");
    const button = document.querySelector(".search-box button");

    if (form) {

        form.addEventListener("submit", () => {

            button.disabled = true;

            button.innerHTML = `
                <i class="fa-solid fa-spinner fa-spin"></i>
                Thinking...
            `;

        });

    }

    // Focus input when page loads
    if (input) {
        input.focus();
    }

    // Press Enter to submit
    if (input) {

        input.addEventListener("keypress", function(e){

            if(e.key === "Enter"){

                form.submit();

            }

        });

    }

    // Smooth fade animation for answer
    const cards = document.querySelectorAll(".user-card, .ai-card");

    cards.forEach((card,index)=>{

        card.style.opacity="0";
        card.style.transform="translateY(25px)";

        setTimeout(()=>{

            card.style.transition="0.5s ease";

            card.style.opacity="1";
            card.style.transform="translateY(0)";

        },200*index);

    });

    // Auto scroll to answer
    const chat = document.querySelector(".chat-area");

    if(chat){

        chat.scrollIntoView({
            behavior:"smooth",
            block:"start"
        });

    }

});