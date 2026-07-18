# Local-build compatibility shim.
#
# github-pages pins liquid 4.0.3, which still calls String#tainted? —
# removed in Ruby 3.2+. Restore taint methods as no-ops so `jekyll serve`
# works on modern Ruby. GitHub Pages builds run in safe mode and ignore
# _plugins, so this file has no effect on the deployed site.
class Object
  unless method_defined?(:tainted?)
    def tainted?
      false
    end

    def taint
      self
    end

    def untaint
      self
    end
  end
end
