# frozen_string_literal: true

module ML
  class Mapper
    attr_reader :x_data, :w_data, :b_data

    def self.call(x_data)
      new(x_data).call
    end

    def initialize(x_data)
      @x_data = x_data
      @w_data = nil
      @b_data = nil
    end

    def call
      @w_data = Array.new(x_data[0].length) { ML::Utils.randn }.map { |e| [e] }
      @b_data = [ML::Utils.randn]

      [w_data, b_data]
    end
  end
end
