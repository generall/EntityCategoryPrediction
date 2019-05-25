<template>
  <div id="app" class="d-flex flex-column h-100">
    <LocaleSwitcher :locales="['en', 'ru']" :show-full="false" class="position-absolute"></LocaleSwitcher>

    <router-view/>

    <!-- FORK Me -->
    <a href="https://github.com/generall/EntityCategoryPrediction">
      <img
        style="position: absolute; top: 0; right: 0; border: 0;"
        src="https://s3.amazonaws.com/github/ribbons/forkme_right_orange_ff7600.png"
        alt="Fork me on GitHub"
      >
    </a>

    <footer class="footer mt-auto py-3 my-footer">
      <div class="container text-center">
        <span class="text-muted" v-html="$t('footer')"></span>
      </div>
    </footer>
  </div>
</template>

<script>
import LocaleSwitcher from "./components/locale_switcher";

export default {
  components: {
    LocaleSwitcher
  }
};
</script>


<style lang="scss">
#nav {
  padding: 30px;
  a {
    font-weight: bold;
    color: #2c3e50;
    &.router-link-exact-active {
      color: #42b983;
    }
  }
}

.my-footer {
  background: #eee;
}

#plus {
  padding: 0 20px 0 20px;
  display: inline-block;
  font-size: 50px;
  vertical-align: top;
  line-height: 100px;
}

.locale-switcher {
  left: 20px;
  top: 20px;

  font-size: 20px;
}

.locale-switcher a {
  color: #08364b;
}

.locale-switcher a:hover {
  color: #515151;
}

html {
  height: 100%;
}

body {
  height: 100%;
}

</style>
