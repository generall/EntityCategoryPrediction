<template>
  <div id="app">
    
    <div id="nav" class="text-center">
      <LocaleSwitcher :locales="['en', 'ru']" :show-full="false" class="position-absolute"> </LocaleSwitcher>
      <router-link to="/">Home</router-link> |
      <router-link to="/demo">Demo</router-link>

    </div>

    <router-view/>

    <!-- FORK Me -->
    <a href="https://github.com/generall/EntityCategoryPrediction"><img style="position: absolute; top: 0; right: 0; border: 0;" src="https://s3.amazonaws.com/github/ribbons/forkme_right_orange_ff7600.png" alt="Fork me on GitHub"></a>

  </div>
</template>

<script>

import LocaleSwitcher from './components/locale_switcher'

export default {
  components: {
    LocaleSwitcher
  }
}
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
#plus {
  padding: 0 20px 0 20px;
  display: inline-block;
  font-size: 50px;
  vertical-align: top;
  line-height: 100px;
}

</style>
