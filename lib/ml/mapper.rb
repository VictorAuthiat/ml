# frozen_string_literal: true

module ML
  class Mapper
    attr_reader :x_data

    def initialize(x_data)
      @x_data = x_data
    end

    def value
      w_data = Array.new(x_data[0].length) { ML::Utils.randn }.map { |e| [e] }
      b_data = [ML::Utils.randn]

      [w_data, b_data]
    end
  end
end
