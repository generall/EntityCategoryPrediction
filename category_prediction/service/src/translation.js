


export default {
  en: {
    "run": "Run",
    "explanation": `
    Paste some text with mentions of any person to the text area below.
    The neural network will try to classify a person based on its mention context.
    This Demo uses <a href="https://spacy.io/">SpaCy</a> library to detect people mentions, after which the neural network is applied.
    There could be many various sources for mentions of a single person, the NN will combine them for more accurate predictions.
    `,
    "prepared_examples": "You can also try on prepared examples:",
    "layers": "Layers",
    "labels": "Predicted labels",
    home: {
      subtitle: "Advanced information extraction",
      demo: "Live Demo",
      subscribe: "Subscribe",
      ner_title: "The next step after  <abbr title='Named Entity Recognition' class='initialism'>NER</abbr>",
      ner_text: ` Mention classification lays between Named Entity Recognition and Entity Linking.
      It provides much more detailed information about an object than NER does.
      At the same time, it does not require storing and maintaining any knowledge base with known objects.
      The Classifier able to work with absolutely new objects never appeared in the train set.`,
      applications_title: "Possible applications",
      applications_text: `Large scale knowledge extraction from news feeds or any web source for specific mentions.
      Additional information for slot filling, chatbots and question answering systems.`,
      nn_arch: "Neural Network Architecture",
      features: "Features",
      feature_one: `The neural net is able to look at up to five mentions simultaneously.
      It combines evidence more efficient than it could be achieved by averaging individual predictions.`,
      feature_two: `Model known
      <b>more than 200</b> general people categories.
      It's also possible to add specific categories without complete model retraining.`,
      coming_soon: "Coming soon",
      lang_support: "Support for other languages",
      other_entities: "Support for other entities",
      organisations: "Organizations",
      events: "Events",
      products: "Products",
    },
    footer: `Interested in collaboration? Contact me on
    <a href='mailto:vasnetsov93@gmail.com?subject=mention+classifier' target="blank">e-mail</a> or
     <a href="tg://resolve?domain=generall93">Telegram</a>`,
  },
  ru: {
    "run": "Предсказать",
    "explanation": `
    Вставьте текст с упоминаниями какого-нибудь человека в текстовое поле ниже.
    Нейросеть попытается классифицировать этого человека на основе контекста его упоминаний.
    Для того, чтобы найти упоминания людей в этой демонстрации используется библиотека <a href="https://spacy.io/">SpaCy</a>.
    Нейронная сеть может использовать сразу несколько упоминаний из разных источников, это сделает предсказание более точным.
    <br/>Сейчас модель работает только с английским языком.
    `,
    "prepared_examples": "Попробуйте начать с подготовленных примеров:",
    "layers": "Слои",
    "labels": "Предсказанные классы",
    home: {
      subtitle: "Продвинутое извлечение информации",
      demo: "Демонстрация",
      subscribe: "Подпишитесь на ",
      ner_title: "Следующий шаг после  <abbr title='Named Entity Recognition' class='initialism'>NER</abbr>",
      ner_text: `Классификация упоминаний является средним между распознаванием именованных сущностей и связыванием сущностей с онтологией.
      Она предоставляет гораздо больше информации об объекте, тем это может NER. В тоже время, для ее работы не требуется наличие какой-либо базы знаний.
      Классификатор может работать с упоминаниями абсолютно незнакомых объектов, про которые ранее ничего не было известно.`,
      applications_title: "Возможные применения",
      applications_text: `Извлечение информации и фактов из потока новостей или любого другого web-источника.
      Улучшение чат-ботов и вопросно-ответных систем за счет извлечения дополнительной информации из текста.`,
      nn_arch: "Архитектура Нейронной Сети",
      features: "Особенности",
      feature_one: `Нейросеть способна учитывать до пяти упоминаний одновременно, что позволяет учитывать все факторы более эффективно, чем этого можно было бы достичь усреднением отдельных предсказаний`,
      feature_two: `Модель знает
      <b>более 200</b> общих категорий для людей.
      Также возможно добавить специфичные категории без полного переобучения модели.`,
      coming_soon: "В разработке",
      lang_support: "Поддержка других языков",
      other_entities: "Поддержка других типов сущностей",
      organisations: "Организации",
      events: "События",
      products: "Товары",
    },
    footer: `Заинтересованы в сотрудничестве? Свяжитесь со мной по
    <a href='mailto:vasnetsov93@gmail.com?subject=mention+classifier' target="blank">e-mail</a> или
     <a href="tg://resolve?domain=generall93">Telegram</a>`,
  }
}
