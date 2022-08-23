let model;

const modelURL = 'http://localhost:5000/model';
const preview = document.getElementById("preview");
const predictButton = document.getElementById("predict");
const clearButton = document.getElementById("clear");
const numberOfFiles = document.getElementById("number-of-files");
const fileInput = document.getElementById('file');

$(document).ready(function(){ 
    $('#file').change(
        function(){
            //Defining the filereader()
            let reader = new FileReader();
            
            //Initialization on Load of the Window.
            reader.onload = function(){
                let dataURL = reader.result;
                $("#selected-image").attr("src",dataURL);
                $("#prediction-list").empty();
            }
            
            let file = $("#file").prop('files')[0];
            
            //reading the file from file object
            reader.readAsDataURL(file);
        });
 });



const predict = async (modelURL) => {
    if (!model) model = await tf.loadLayersModel(modelURL);


    let image= $('#selected-image').get(0);
    console.log(image);

    let tensor = tf.browser.fromPixels(image)
                    .resizeNearestNeighbor([160,160])
                    .mul(1/127.5)
                    .toFloat().sub(1)
                    .expandDims();

    // comment: model_classification_8_types_of_card
    // console.log("model_classification_8_types_of_card")
    // const list_class = ['back_new_national_id_ver_1', 'back_new_national_id_ver_2', 'back_old_national_id', 'font_driver_license', 
    //                 'font_new_national_id_ver_1', 'font_new_national_id_ver_2', 'font_old_national_id', 'passport']

    // comment: model_document_classification_4_types
    console.log("model_document_classification_4_types")
    const list_class = ['other', 'registration_book', 'old_national_id', 'vehicle_registration']

    const prediction = model.predict(tensor)
    const label = prediction.argMax(axis = 1).dataSync()[0];
    const class_of_label = list_class[label]


    console.log("label")
    console.log(label)
    console.log(class_of_label)
};

fileInput.addEventListener("change", () => numberOfFiles.innerHTML = "Selected " + fileInput.files.length + " files", false);
predictButton.addEventListener("click", () => predict(modelURL));
clearButton.addEventListener("click", () => preview.innerHTML = "");
