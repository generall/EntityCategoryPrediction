<template>
  <div class="locale-switcher">
    <span v-for="locale in locales"
      :key="locale.id"><a
      class="locale-link"
      @click="setLocale(locale)"
      :class="{ 'is-current': locale === activeLocale }"
      href="javascript:void(0)"
    >{{ getLanguageString(locale) }}</a>&nbsp;</span>
  </div>
</template>

<script>
import Vue from "vue";
// Restore locale from cookie, if it was set
import VueCookie from "vue-cookie";
Vue.use(VueCookie);

const localeStrings = {
  en: "English",
  ru: "Russian"
};

Vue.config.lang = VueCookie.get("locale") || "en";
console.log(
  "Locale from cookie = " +
    Vue.config.lang +
    ": language = " +
    localeStrings[Vue.config.lang]
);

export default {
  props: {
    locales: {
      type: Array,
      default: ["en"]
    },
    showFull: Boolean,
    dropup: Boolean,
    locLabel: {
      type: String,
      default: ""
    }
  },
  mounted: function() {
    this.setLocale(Vue.config.lang)
  },
  data: function() {
    return {
      activeLocale: Vue.config.lang
    };
  },
  computed: {
    dropdownLbl: function() {
      return this.locLabel ? this.locLabel : this.activeLocale;
    }
  },
  methods: {
    setLocale: function(locale) {
      Vue.config.lang = locale;
      this.activeLocale = locale;
      this.$cookie.set("locale", locale);
      this.$i18n.locale = Vue.config.lang;
      console.log(
        "New locale = " +
          Vue.config.lang +
          ": language = " +
          localeStrings[Vue.config.lang]
      );
    },
    getLanguageString: function(locale) {
      return this.showFull ? localeStrings[locale] : locale;
    }
  }
};
</script>

<style>
.is-current {
  font-weight: bold;
}

#nav a.locale-link {
  font-weight: normal;
}
.is-current {
  text-decoration: underline;
}
</style>
